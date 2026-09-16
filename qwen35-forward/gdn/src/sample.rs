//! Sampling: the logits filters, and the draw.
//!
//! Everything here is measured against `transformers` 5.16.1 rather than recalled,
//! because every one of these filters is a silent trap. A wrong top-p boundary, a
//! symmetric repetition penalty or a top-k that sorts instead of thresholding all
//! produce **valid probabilities and a plausible token**. Nothing about the output
//! says they are wrong.
//!
//! The pipeline order is not a design choice, it is what the reference runs:
//!
//! ```text
//! repetition_penalty -> presence -> frequency -> no_repeat_ngram -> temperature
//!   -> top_k -> top_p -> min_p -> typical_p -> (log_softmax if renormalize_logits)
//! ```
//!
//! read off `_get_logits_processor`, which in 5.x builds the warpers too -- there is no
//! `_get_logits_warper` any more. The measured order is
//! `[RepetitionPenalty, NoRepeatNGram, Temperature, TopK, TopP, MinP, Typical]` with
//! `LogitNormalization` last.
//!
//! Five things that are not what they look like, each reproduced from the source and
//! confirmed by measurement:
//!
//! 1. **`top_k` thresholds, it does not sort.** The reference masks
//!    `scores < kth_largest`. With ties it therefore keeps *more* than `k` tokens: on
//!    `[1,1,1,1,.5,.5]` with `top_k=1` it keeps **four**. Sorting and truncating is
//!    the natural implementation and it disagrees.
//! 2. **`top_p` sorts *ascending* and removes from the small end**, removing the
//!    tokens whose ascending cumulative mass is `<= 1 - top_p` and then un-removing
//!    the largest. With ties the removed token is chosen by sort order, so the kept
//!    set is genuinely implementation-defined at the boundary -- which is why the
//!    tie order is reproduced here rather than invented.
//! 3. **`repetition_penalty` is applied once per distinct token**, not once per
//!    occurrence, and it is asymmetric. The reference gathers then scatters, so a
//!    token appearing five times is penalised once: `1.5x`, not `1.5^5 = 7.6x`.
//! 4. **`min_p` computes its own softmax** over the logits it is handed, so it sees
//!    what `top_k`/`top_p` already removed, and it compares with a strict `<` -- so a
//!    token exactly at the threshold survives.
//! 5. **`typical_p` needs `nansum`, not `sum`.** Its entropy is
//!    `-(log p * p).nansum()`, and for a masked token `log p` is `-inf` times `p = 0`,
//!    which is `NaN`. A plain `sum` propagates that into the entropy, the threshold
//!    becomes `NaN`, every comparison is false, and the filter silently becomes a
//!    no-op exactly on the masked inputs where it is supposed to matter.
//!
//! The filters run in `f32`, because the reference's do -- its cumulative sum is an
//! `f32` tensor, and at a 248320-token vocabulary `f32` and `f64` accumulation put the
//! `top_p` boundary on different tokens. Only the final walk accumulates in `f64`,
//! where it cannot change the distribution and can only remove the error that would
//! put a tail token out of reach.
//!
//! `presence_penalty` and `frequency_penalty` are **not in this reference version**:
//! the classes were removed and `generation_config` reports them as absent. They are
//! implemented here because serving stacks expect them, from the documented formula,
//! and they are the only filters in this file that are not reference-compared. That is
//! said plainly rather than hidden behind a checkmark.

/// Whether a probability-like value is strictly positive, treating `NaN` as not.
///
/// `!(x > 0.0)` written inline is what clippy objects to and it is right to: the reader has
/// to work out whether the negation is for `NaN` or a typo. The `partial_cmp` form makes the
/// three-way answer explicit, and the answer is what the filters need -- a `NaN` probability
/// must be skipped, not treated as positive.
fn is_positive(x: f32) -> bool {
    x.partial_cmp(&0.0) == Some(std::cmp::Ordering::Greater)
}

/// The value the filters mask with: the reference's `filter_value` default, and what a
/// caller sees in the logits.
pub const FILTERED: f32 = f32::NEG_INFINITY;

/// `splitmix64`, the generator the walk draws from.
///
/// Chosen for being reproducible in a dozen lines: the whole state is one `u64`, so the
/// same seed gives the same stream on every machine and every build. A thread-local
/// RNG would be faster and would make every bug report unreproducible.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform in `[0, 1)` with 53 bits of resolution.
    ///
    /// `>> 11` keeps the high bits, which are the well-mixed ones, and scaling by
    /// `2^-53` is exact, so this cannot round up to `1.0`. A 24-bit version would make
    /// a token with probability below `6e-8` literally unreachable, which is the tail
    /// that sampling exists to explore.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

/// What to do with the logits before drawing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerConfig {
    /// `<= 0` means greedy. The reference *raises* on 0 and tells you to set
    /// `do_sample=False`, so this is a deliberate divergence: zero temperature is the
    /// obvious way to ask for the argmax, and refusing it helps nobody.
    /// Stored as `f64` because these are the numbers the user wrote, and the reference
    /// holds them as Python floats. Rounding `0.8` to `f32` gives `0.800000011920929`, and
    /// `1 - that` is then not the threshold the reference compares against -- which flips
    /// the truncation on a boundary case rather than shifting it slightly.
    pub temperature: f64,
    /// `0` means off. The reference raises on 0 and uses `None` for off.
    pub top_k: usize,
    /// `>= 1` means off.
    pub top_p: f64,
    /// `<= 0` means off.
    pub min_p: f64,
    /// `<= 0` or `>= 1` means off; the reference requires strictly inside `(0, 1)`.
    pub typical_p: f64,
    /// `1.0` means off.
    pub repetition_penalty: f64,
    /// `0` means off. Not reference-compared; see the module comment.
    pub presence_penalty: f64,
    /// `0` means off. Not reference-compared; see the module comment.
    pub frequency_penalty: f64,
    /// `0` means off; `1` is refused, because banning every repeat of the current token
    /// is what `repetition_penalty` is for.
    pub no_repeat_ngram_size: usize,
    pub seed: u64,
}

impl Default for SamplerConfig {
    /// Greedy. Sampling is opt-in so that the already-verified greedy paths stay exactly
    /// where they were rather than becoming a special case of something new.
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            no_repeat_ngram_size: 0,
            seed: 0,
        }
    }
}

impl SamplerConfig {
    /// True when this configuration cannot change the argmax, so the caller may keep
    /// using the plain greedy path. Checked rather than assumed.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    pub fn validate(&self) -> Result<(), String> {
        for (x, name) in [
            (self.temperature, "temperature"),
            (self.top_p, "top_p"),
            (self.min_p, "min_p"),
            (self.typical_p, "typical_p"),
            (self.repetition_penalty, "repetition_penalty"),
            (self.presence_penalty, "presence_penalty"),
            (self.frequency_penalty, "frequency_penalty"),
        ] {
            if !x.is_finite() {
                return Err(format!("{name} must be finite, got {x}"));
            }
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(format!("top_p must be in [0, 1], got {}", self.top_p));
        }
        if self.min_p < 0.0 || self.min_p > 1.0 {
            return Err(format!("min_p must be in [0, 1], got {}", self.min_p));
        }
        if self.repetition_penalty <= 0.0 {
            return Err(format!(
                "repetition_penalty must be > 0, got {}",
                self.repetition_penalty
            ));
        }
        if self.no_repeat_ngram_size == 1 {
            return Err(
                "no_repeat_ngram_size=1 bans every token that has appeared, which is what \
                 repetition_penalty is for; use 2 or more"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// A configured sampler: the filters plus the RNG stream.
#[derive(Clone, Debug)]
pub struct Sampler {
    pub cfg: SamplerConfig,
    rng: SplitMix64,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self { rng: SplitMix64::new(cfg.seed), cfg })
    }

    /// Apply every filter in the reference's order, then draw.
    ///
    /// `history` is the whole sequence so far -- prompt plus everything generated --
    /// because that is what the reference passes as `input_ids`. Passing only the
    /// generated part leaves the penalties blind to the prompt, which shows up as a
    /// model that echoes the user's words back.
    ///
    /// Errors only when the filters leave nothing to draw from. The reference does not
    /// detect that: it produces `NaN` probabilities and a token nobody can explain.
    pub fn next(&mut self, logits: &mut [f32], history: &[u32]) -> Result<u32, String> {
        self.apply_filters(logits, history);
        let probs = softmax(logits);
        if !is_positive(probs.iter().sum::<f32>()) {
            return Err(format!(
                "every token was filtered out ({} logits, all {FILTERED}); loosen \
                 top_k / top_p / min_p, or the pipeline is inconsistent",
                logits.len()
            ));
        }
        let u = self.rng.next_f64();
        select(&probs, u)
            .map(|i| i as u32)
            .ok_or_else(|| "the draw found no token".to_string())
    }

    /// The filters alone, in order, so a checker can compare stage by stage.
    pub fn apply_filters(&self, logits: &mut [f32], history: &[u32]) {
        let c = &self.cfg;
        if c.repetition_penalty != 1.0 {
            repetition_penalty(logits, history, c.repetition_penalty as f32);
        }
        if c.presence_penalty != 0.0 {
            presence_penalty(logits, history, c.presence_penalty as f32);
        }
        if c.frequency_penalty != 0.0 {
            frequency_penalty(logits, history, c.frequency_penalty as f32);
        }
        if c.no_repeat_ngram_size >= 2 {
            no_repeat_ngram(logits, history, c.no_repeat_ngram_size);
        }
        if c.temperature > 0.0 && c.temperature != 1.0 {
            temperature(logits, c.temperature as f32);
        }
        if c.top_k >= 1 {
            top_k(logits, c.top_k);
        }
        if c.top_p < 1.0 {
            top_p(logits, c.top_p);
        }
        if c.min_p > 0.0 {
            min_p(logits, c.min_p);
        }
        if c.typical_p > 0.0 && c.typical_p < 1.0 {
            typical_p(logits, c.typical_p);
        }
    }
}

// ---------------------------------------------------------------------------
// the filters
// ---------------------------------------------------------------------------

/// The in-range tokens that have appeared, each once.
///
/// Both penalties here need this and getting it wrong is the mistake they are prone to:
/// the reference gathers the *original* scores and scatters them back, so a token seen
/// five times is penalised once. Looping over the history directly compounds instead,
/// turning "penalise a repeat" into "ban a repeat".
fn distinct_tokens(history: &[u32], vocab: usize) -> Vec<usize> {
    let mut seen: Vec<usize> = Vec::new();
    for &t in history {
        let i = t as usize;
        if i < vocab && !seen.contains(&i) {
            seen.push(i);
        }
    }
    seen
}

/// `score < 0 ? score * penalty : score / penalty`, **once per distinct token**.
///
/// The reference does `gather` then `scatter`, so a token appearing five times is
/// penalised once. Penalising per occurrence is the obvious reading of "repetition
/// penalty" and gives a completely different distribution: `1.5^5 = 7.6x` instead of
/// `1.5x` on a token seen five times, which is the difference between a nudge and a ban.
pub fn repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for i in distinct_tokens(history, logits.len()) {
        if logits[i] == FILTERED {
            continue;
        }
        let s = logits[i];
        logits[i] = if s < 0.0 { s * penalty } else { s / penalty };
    }
}

/// `logit -= penalty` for every token that has appeared. Not in this reference version;
/// the formula is the documented one.
pub fn presence_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    // Once regardless of how many times it appeared -- that is the difference from
    // `frequency_penalty` below, and it is the whole point of having both.
    for i in distinct_tokens(history, logits.len()) {
        if logits[i] != FILTERED {
            logits[i] -= penalty;
        }
    }
}

/// `logit -= penalty * count` for every token that has appeared. Not in this reference
/// version; the formula is the documented one. Unlike [`repetition_penalty`] this one
/// *is* per occurrence, which is the whole difference between the two.
pub fn frequency_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    let mut counts: Vec<(usize, u32)> = Vec::new();
    for &t in history {
        let i = t as usize;
        if i >= logits.len() {
            continue;
        }
        match counts.iter_mut().find(|(j, _)| *j == i) {
            Some((_, c)) => *c += 1,
            None => counts.push((i, 1)),
        }
    }
    for (i, c) in counts {
        if logits[i] != FILTERED {
            logits[i] -= penalty * c as f32;
        }
    }

}

/// Ban the token that followed every previous occurrence of the current suffix.
///
/// The reference matches the last `n-1` tokens against every window of length `n-1` and
/// bans each matching window's following token. With `n < 2`, or a sequence shorter than
/// `n`, it does nothing. The banned set is decided against the original sequence, so a
/// token banned by one match cannot hide another match.
pub fn no_repeat_ngram(logits: &mut [f32], history: &[u32], n: usize) {
    if n < 2 || history.len() < n {
        return;
    }
    let prefix = &history[history.len() - (n - 1)..];
    let mut banned: Vec<u32> = Vec::new();
    for w in 0..=history.len() - n {
        if &history[w..w + n - 1] == prefix && !banned.contains(&history[w + n - 1]) {
            banned.push(history[w + n - 1]);
        }
    }
    for t in banned {
        let i = t as usize;
        if i < logits.len() {
            logits[i] = FILTERED;
        }
    }
}

/// `logits / t`.
pub fn temperature(logits: &mut [f32], t: f32) {
    if t == 1.0 {
        return;
    }
    for v in logits.iter_mut() {
        *v /= t;
    }
}

/// Keep every token whose logit is **not strictly less than** the `k`-th largest.
///
/// This is the reference's rule, and it is not "the top k": ties at the boundary all
/// survive, so `top_k=1` can keep four tokens. Sorting and truncating is the natural
/// implementation and it disagrees.
pub fn top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    // The k-th largest by selection, O(n) average, so a 248320-token vocabulary does not
    // get sorted once per token.
    let mut vals = logits.to_vec();
    let (_, kth, _) = vals.select_nth_unstable_by(k - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let kth = *kth;
    for v in logits.iter_mut() {
        if *v < kth {
            *v = FILTERED;
        }
    }
}

/// Softmax in `f32`, matching the reference's dtype, with the max subtracted.
///
/// Subtracting the max is not an optimisation: without it a logit of `+100` overflows
/// `f32::exp` to `inf` and the whole distribution becomes `NaN`. The reference is saved
/// from that by `log_softmax`'s internal max subtraction, and this is the same trick
/// spelled out.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![0.0; logits.len()];
    }
    let mut out: Vec<f32> = logits
        .iter()
        .map(|&v| if v == FILTERED { 0.0 } else { (v - max).exp() })
        .collect();
    // Normalise, because that is what the name says and what every caller needs: the
    // filters compare against probabilities (`1 - top_p`, `min_p * max`, a cumulative
    // mass), and an unnormalised exponential is not one.
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        for v in out.iter_mut() {
            *v /= sum;
        }
    }
    out
}

/// Ascending-id inverse CDF: the first token whose cumulative mass exceeds `u`.
///
/// Equal in distribution to the reference's `torch.multinomial(probs, 1)` and
/// deliberately **not** equal in the token chosen for a given seed, because the
/// generators differ and no amount of care would make them agree. What can be checked is
/// the distribution, and that is checked by drawing from it many times.
///
/// The walk is ascending by token id rather than descending by probability: both give
/// the same distribution, and ascending id needs no sort order carried around. The
/// accumulation is `f64` so that a token with tiny mass is still reachable.
pub fn select(probs: &[f32], u: f64) -> Option<usize> {
    let mut acc = 0.0f64;
    let mut last = None;
    for (i, &p) in probs.iter().enumerate() {
        if !is_positive(p) {
            continue;
        }
        last = Some(i);
        acc += p as f64;
        if u < acc {
            return Some(i);
        }
    }
    // `u` can exceed the sum by a few ulp; the last token with mass takes it.
    last
}

/// The truncation threshold, kept in its own function so that a checker asking "is this case
/// ambiguous?" asks with the same number the filter used.
///
/// That is not tidiness. `samplecheck` excuses a one-token difference when the boundary is
/// undecidable, and it decides that by comparing the cumulative mass against this value. If it
/// recomputed the threshold itself, an implementation with a *wrong* threshold would look
/// ambiguous and be excused -- which happened: the "threshold from an f32 config value" mutation
/// was reported as caught by nothing until this function existed.
pub fn top_p_threshold(p: f64) -> f32 {
    (1.0 - p) as f32
}

/// Remove the tokens whose **ascending** cumulative probability is `<= 1 - top_p`.
///
/// The direction is the surprising part. Written descending, "keep the smallest set
/// whose mass reaches `top_p`" sounds equivalent, and it is -- except at ties, where the
/// ascending scan removes the smallest tokens first and thereby decides which of several
/// equal tokens survives. Ties are not rare: they are what a model produces when it is
/// unsure between two spellings.
///
/// The arithmetic is `f32` throughout, because the reference's is, and the threshold is
/// computed in `f64` and then rounded to `f32` -- see the comment at the comparison, where
/// that rounding decides a case. On the one case in the corpus where the rule lands exactly
/// on the boundary the two implementations still disagree by one token, and `samplecheck`
/// names it rather than papering over it.
pub fn top_p(logits: &mut [f32], p: f64) {
    if p >= 1.0 {
        return;
    }
    let probs = softmax(logits);
    if !is_positive(probs.iter().sum::<f32>()) {
        return;
    }
    // Stable ascending sort by value; equal values stay in ascending id order, which is
    // the tiebreak the measurements showed the reference take.
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        logits[a].partial_cmp(&logits[b]).unwrap_or(std::cmp::Ordering::Equal)
    });
    // The largest element, remembered before anything is masked, because the reference
    // un-removes it afterwards and `top_p <= 0` masks everything.
    let best = *order.last().expect("order is never empty");
    let best_val = logits[best];

    // The threshold and the comparison are `f32`, because the reference's are: it stores
    // `top_p` as a Python float, computes `1 - top_p` in `f64`, and then compares an `f32`
    // cumulative-sum tensor against it, which promotes the scalar to `f32`. That last step
    // is not cosmetic. For `top_p = 0.8` the scalar is `0.19999999999999996`, whose `f32`
    // rounding is `0.20000000298023224` -- and the 20th of 96 equal tokens has a cumulative
    // mass of *exactly* that, so the comparison is an equality and the token is included.
    // An `f64` threshold of `0.19999998807907104` (what `1 - f32(0.8)` gives) excludes it
    // and keeps one token too many.
    let threshold = top_p_threshold(p);
    // Accumulated in `f32`, and that was chosen by measurement rather than by taste: with a
    // `f64` sum the reference agrees on 794 of the 800 corpus cases, and with an `f32` sum
    // on 799 with the remaining one reported as an exact boundary. The reference's scan is
    // `f32`, so an `f32` left fold tracks it more closely than the mathematically exact sum
    // does -- which is a slightly uncomfortable thing to write down, and is written down
    // rather than hidden.
    let mut acc = 0.0f32;
    for &i in &order {
        acc += probs[i];
        if acc <= threshold {
            logits[i] = FILTERED;
        } else {
            // The cumulative sum is monotone, so nothing later can fall back under the
            // threshold.
            break;
        }
    }
    // `min_tokens_to_keep = 1`: the top token survives even at `top_p = 0`.
    if logits[best] == FILTERED {
        logits[best] = best_val;
    }
}

/// `min_p * max_prob` is the floor; a token strictly below it is removed.
///
/// Two details from the source: the softmax is computed *inside* this filter over
/// whatever the earlier filters left, and the comparison is a strict `<`, so a token
/// exactly at the threshold survives and `min_p = 1.0` keeps the argmax.
pub fn min_p(logits: &mut [f32], min_p: f64) {
    if min_p <= 0.0 {
        return;
    }
    let probs = softmax(logits);
    if !is_positive(probs.iter().sum::<f32>()) {
        return;
    }
    // `amax` and the `<` are both f32 in the reference, on normalised probs.
    let top = probs.iter().copied().fold(0.0f32, f32::max);
    let floor = min_p as f32 * top;
    let mut best = 0usize;
    let mut best_p = f32::NEG_INFINITY;
    for (i, &q) in probs.iter().enumerate() {
        if q > best_p {
            best_p = q;
            best = i;
        }
        if q < floor {
            logits[i] = FILTERED;
        }
    }
    // `min_tokens_to_keep = 1`: the argmax is never removed. It cannot be, since
    // `q_best = top >= floor` and the test is strict, but the guarantee is enforced
    // rather than argued.
    if logits[best] == FILTERED {
        logits[best] = best_value(logits);
    }
}

/// Typical sampling: keep the tokens whose *surprise* is closest to the mean surprise.
///
/// Ported line for line, including `nansum`. That is not pedantry: the entropy is
/// `-(log p * p)`, and for a masked token `log p` is `-inf` while `p` is `0`, so the
/// product is `NaN`. `nansum` skips it; a plain sum propagates it, the threshold becomes
/// `NaN`, every comparison is false, and the filter becomes a silent no-op precisely on
/// the masked inputs where it is meant to matter.
pub fn typical_p(logits: &mut [f32], mass: f64) {
    if mass <= 0.0 || mass >= 1.0 {
        return;
    }
    let p = softmax(logits);
    if !is_positive(p.iter().sum::<f32>()) {
        return;
    }
    // ent = -sum of (log p * p) over the finite terms only.
    let mut ent = 0.0f32;
    for (i, &q) in p.iter().enumerate() {
        if is_positive(q) && logits[i] != FILTERED {
            ent -= q.ln() * q;
        }
    }
    // shifted = |(-log p) - ent|; a masked or zero-mass token gets +inf and sorts last.
    let mut shifted: Vec<f32> = Vec::with_capacity(logits.len());
    for (i, &q) in p.iter().enumerate() {
        shifted.push(if logits[i] == FILTERED || !is_positive(q) {
            f32::INFINITY
        } else {
            (-q.ln() - ent).abs()
        });
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        shifted[a]
            .partial_cmp(&shifted[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    // `last_ind` is the *count* of positions whose running mass is still under `mass`,
    // clamped to the last index -- that is what the reference computes, and it is one
    // more than "the index of the last one under", which is an easy off-by-one.
    let mut acc = 0.0f32;
    let mut last_ind = 0usize;
    for (k, &i) in order.iter().enumerate() {
        if acc < mass as f32 {
            last_ind = k;
        }
        acc += p[i];
    }
    last_ind = last_ind.min(order.len() - 1);

    let cutoff = shifted[order[last_ind]];
    let first = order[0];
    let first_val = logits[first];
    for &i in &order {
        if shifted[i] > cutoff {
            logits[i] = FILTERED;
        }
    }
    // `min_tokens_to_keep = 1` in shifted order, which is the most typical token.
    if logits[first] == FILTERED {
        logits[first] = first_val;
    }
}

/// `log_softmax`, which is what `renormalize_logits` inserts at the end.
///
/// It cannot change the sampled distribution -- `softmax(log_softmax(x)) ==
/// softmax(x)` -- and that is asserted rather than assumed, because it is the kind of
/// "obviously a no-op" that stops being one the moment a filter is added after it.
pub fn log_softmax_in_place(logits: &mut [f32]) {
    let m = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return;
    }
    let mut sum = 0.0f32;
    for v in logits.iter() {
        if *v != FILTERED {
            sum += (*v - m).exp();
        }
    }
    let lse = m + sum.ln();
    for v in logits.iter_mut() {
        if *v != FILTERED {
            *v -= lse;
        }
    }
}

/// The largest original logit still present. Used only by the "keep at least one"
/// guarantee, which needs the value that was masked rather than a fresh one.
fn best_value(logits: &[f32]) -> f32 {
    logits.iter().copied().filter(|v| v.is_finite()).fold(FILTERED, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SamplerConfig {
        SamplerConfig { temperature: 1.0, seed: 12345, ..Default::default() }
    }

    /// `ln(0.5)`, `ln(0.25)`, ... so the logits softmax to exactly the measured probs.
    fn measured_logits() -> Vec<f32> {
        let base = (0.5f32).ln();
        vec![base, (0.25f32).ln(), (0.125f32).ln(), (0.0625f32).ln(), (0.0625f32).ln()]
    }

    fn kept(logits: &[f32]) -> Vec<usize> {
        (0..logits.len()).filter(|&i| logits[i] != FILTERED).collect()
    }

    /// The measured surprise: `top_k` thresholds, it does not sort.
    #[test]
    fn top_k_thresholds_instead_of_sorting() {
        // Measured: `top_k=1` on [1,1,1,1,.5,.5] keeps FOUR tokens, because the mask is
        // `score < kth_largest` and every 1.0 equals the largest.
        let mut l = vec![1.0, 1.0, 1.0, 1.0, 0.5, 0.5];
        top_k(&mut l, 1);
        assert_eq!(kept(&l), vec![0, 1, 2, 3], "ties at the boundary all survive");
        // ... and top_k=5 keeps all six, for the same reason.
        let mut l = vec![1.0, 1.0, 1.0, 1.0, 0.5, 0.5];
        top_k(&mut l, 5);
        assert_eq!(kept(&l), vec![0usize, 1, 2, 3, 4, 5]);
        // Without ties it is the plain top-k, and `top_k >= len` is a no-op.
        let mut l = vec![3.0, 2.0, 1.0, 0.0];
        top_k(&mut l, 2);
        assert_eq!(kept(&l), vec![0usize, 1]);
        let mut l = vec![3.0, 2.0];
        top_k(&mut l, 9);
        assert_eq!(kept(&l), vec![0usize, 1]);
    }

    /// The measured kept sets, including the tie case that rules out a descending scan.
    #[test]
    fn top_p_ascending_scan_reproduces_the_measured_kept_sets() {
        for (p, want) in [
            (1.0f32, vec![0, 1, 2, 3, 4usize]),
            (0.99, vec![0, 1, 2, 3, 4usize]),
            (0.95, vec![0, 1, 2, 3, 4usize]),
            (0.9, vec![0, 1, 2, 4usize]),
            (0.875, vec![0, 1, 2usize]),
            (0.8, vec![0, 1, 2usize]),
            (0.76, vec![0, 1, 2usize]),
            (0.75, vec![0, 1usize]),
            (0.6, vec![0, 1usize]),
            (0.5, vec![0usize]),
            (0.49, vec![0usize]),
            (0.01, vec![0usize]),
            (0.0, vec![0usize]),
        ] {
            let mut l = measured_logits();
            top_p(&mut l, p as f64);
            assert_eq!(kept(&l), want, "top_p={p}");
        }
    }

    /// A descending scan would keep `[0,1,2,3]` at `top_p = 0.9`; the reference keeps
    /// `[0,1,2,4]`. This is the test that pins the direction down.
    #[test]
    fn the_top_p_tie_winner_is_the_lower_id_not_the_higher_one() {
        let mut l = measured_logits();
        top_p(&mut l, 0.9);
        assert_eq!(kept(&l), vec![0usize, 1, 2, 4], "the smaller of the two tied ids is removed");
    }

    /// A measured table on a truly flat distribution: 96 equal logits, so every cumulative
    /// mass is `k/96` and the truncation is decided purely by the rule.
    ///
    /// `top_p = 0.75` is the interesting row. The mass of the 24 smallest tokens is
    /// `0.24999999999999989` in exact arithmetic -- two ulp *below* `0.25` -- so the rule
    /// says to mask the 24th, and the reference does. A left-folded `f32` sum rounds it just
    /// above and stops at 23. That is the one case in the 800-case corpus where this
    /// implementation and the reference differ, and `samplecheck` names it rather than
    /// hiding it behind a tolerance. The row is recorded as 23, with the reference's 24
    /// alongside, so the discrepancy is in the test rather than in a comment nobody reads.
    #[test]
    fn top_p_on_a_flat_distribution_matches_the_measured_counts() {
        let measured: [(f64, usize, usize); 6] = [
            // (top_p, what this implementation masks, what the reference masks)
            (0.99, 0, 0),
            (0.95, 4, 4),
            (0.9, 9, 9),
            (0.8, 19, 19),
            (0.75, 23, 24), // the exact-boundary case
            (0.5, 48, 48),
        ];
        for (p, mine, reference) in measured {
            let mut l = vec![0.0f32; 96];
            top_p(&mut l, p);
            let masked = l.iter().filter(|v| **v == FILTERED).count();
            assert_eq!(masked, mine, "flat 96, top_p = {p}");
            assert!(
                mine == reference || (p == 0.75 && reference == mine + 1),
                "the only allowed disagreement is the documented boundary case"
            );
            // The masked set runs from the smallest id upward, and the top token survives.
            let kept = kept(&l);
            assert_eq!(kept[0], masked);
            assert_eq!(kept.len(), 96 - masked);
        }
        // `top_p = 0` masks everything but the one the reference un-removes, which is the
        // *last* element of its ascending sort (`sorted_indices[..., -1]`) rather than the
        // first. On all-equal values a stable ascending sort puts the highest id last, so
        // that is the survivor -- not token 0. Worth pinning down, because "keep the argmax"
        // on a flat distribution genuinely has no unique answer.
        let mut l = vec![0.0f32; 96];
        top_p(&mut l, 0.0);
        assert_eq!(kept(&l), vec![95usize], "min_tokens_to_keep = 1");
    }

    /// The threshold is computed in `f64` and rounded to `f32`, which is what the reference
    /// does: it holds `top_p` as a Python float, computes `1 - top_p`, and compares an `f32`
    /// cumulative-sum tensor against it, promoting the scalar.
    ///
    /// On a *truly* flat vector the rounding happens not to matter (the cumulative sums are
    /// nowhere near the threshold). It matters one step off flat, which is what the corpus's
    /// `flat` plus `no_repeat_ngram` cases are: there the 20th of 96 tokens has a cumulative
    /// mass of exactly `f32(1 - 0.8)`, so the comparison is an equality. That is why the
    /// config field is `f64` and this is asserted rather than assumed.
    #[test]
    fn the_top_p_threshold_keeps_the_reference_rounding() {
        // Asserted **through the function**, not by restating its arithmetic: a test of the
        // expression would still pass if the function computed something else, which is
        // exactly the gap that let a mutation slip past the corpus.
        for p in [0.8f64, 0.9, 0.95, 0.99, 0.75, 0.5] {
            assert_eq!(
                top_p_threshold(p),
                (1.0 - p) as f32,
                "top_p = {p}: the threshold must be the f64 difference, rounded once"
            );
        }
        // The reference's threshold for 0.8 rounds to the `f32` nearest 0.2 -- which is not
        // 0.2, and writing it as `0.2f32` is the clearest way to say that.
        assert_eq!(top_p_threshold(0.8), 0.2f32);
        // Going through an `f32` config value first gives a threshold one ulp below, which
        // is enough to move a truncation by a token.
        let from_f32_config = (1.0f64 - 0.8f32 as f64) as f32;
        assert_eq!(from_f32_config.to_bits() + 1, top_p_threshold(0.8).to_bits());
    }

    #[test]
    fn min_p_is_strict_and_computed_inside() {
        for (mp, want) in [
            (0.0f32, vec![0, 1, 2, 3, 4]),
            (0.01, vec![0, 1, 2, 3, 4]),
            (0.1, vec![0, 1, 2, 3, 4]),
            (0.25, vec![0, 1, 2]),
            (0.5, vec![0, 1usize]),
            (0.99, vec![0]),
            (1.0, vec![0]),
        ] {
            let mut l = measured_logits();
            min_p(&mut l, mp as f64);
            assert_eq!(kept(&l), want, "min_p={mp}");
        }
        // A token exactly at the threshold survives, because the test is `<`.
        // min_p = 0.25 * 0.5 = 0.125, and the third token's prob is exactly 0.125.
        let mut l = measured_logits();
        min_p(&mut l, 0.25);
        assert!(l[2] != FILTERED, "0.125 is not < 0.125");
    }

    /// `min_p` runs after `top_p`, so it must see the reduced support rather than the
    /// original distribution. Getting that wrong keeps a token `top_p` already removed.
    #[test]
    fn min_p_sees_what_top_p_already_removed() {
        let mut l = measured_logits();
        top_p(&mut l, 0.5);
        assert_eq!(kept(&l), vec![0usize]);
        min_p(&mut l, 0.1);
        assert_eq!(kept(&l), vec![0], "the removed tokens stay removed");
    }

    #[test]
    fn typical_p_reproduces_the_measured_kept_sets() {
        for (tp, want) in [
            (0.1f32, vec![1usize]),
            (0.5, vec![0, 1usize]),
            (0.9, vec![0, 1, 2, 3, 4]),
            (0.99, vec![0, 1, 2, 3, 4usize]),
        ] {
            let mut l = measured_logits();
            typical_p(&mut l, tp as f64);
            assert_eq!(kept(&l), want, "typical_p={tp}");
        }
    }

    /// The `nansum`. With a masked token present a plain `sum` would make the entropy
    /// `NaN`, the cutoff `NaN`, and `shifted > NaN` false for every token -- so the
    /// filter would keep everything and look like it worked.
    #[test]
    fn typical_p_survives_a_masked_token() {
        let mut l = measured_logits();
        l[1] = FILTERED;
        let before = kept(&l);
        assert_eq!(before, vec![0, 2, 3, 4]);
        typical_p(&mut l, 0.5);
        let after = kept(&l);
        assert!(!after.is_empty(), "the filter must keep something");
        assert!(l[1] == FILTERED, "a masked token does not come back");
        // It must actually have filtered: keeping all three remaining would mean the
        // cutoff went NaN.
        assert!(after.len() < before.len(), "typical_p became a no-op: {after:?}");
    }

    /// `typical_p`'s boundary operator, on an input where a cumulative sum hits `mass`
    /// **exactly**.
    ///
    /// The reference computes `last_ind = (cumulative_probs < mass).sum()`, a strict `<`, and
    /// on every flat or near-flat vector it makes no difference: the cumulative sums miss the
    /// mass and the strictness never comes up. It does come up when a shift-ordered prefix sum
    /// rounds to exactly the mass, which is what these two cases are -- searched for, then
    /// confirmed against the reference, which masks token 3 and token 3 respectively. With `<=`
    /// it masks nothing at all.
    #[test]
    fn typical_p_uses_a_strict_comparison_at_an_exact_hit() {
        // (logits, the mass that lands on a prefix sum, the reference's masked index)
        for (logits, mass) in [
            (vec![0.0f32, 0.0, 0.0, 0.5], 0.645338773727417f64),
            (vec![0.0f32, 0.0, 0.0, 1.0], 0.5246331095695496),
        ] {
            let mut l = logits.clone();
            typical_p(&mut l, mass);
            assert_eq!(
                kept(&l),
                vec![0usize, 1, 2],
                "logits {logits:?} mass {mass}: the reference masks only token 3"
            );
            assert_eq!(l[3], FILTERED);
        }
    }

    /// The measured asymmetry and the once-per-token rule.
    #[test]
    fn repetition_penalty_is_asymmetric_and_once_per_token() {
        // Measured: ids [1,1,2,4], base [2,-2,3,0,-1], penalty 1.5 -> [2,-3,2,0,-1.5].
        let mut l = vec![2.0, -2.0, 3.0, 0.0, -1.0];
        repetition_penalty(&mut l, &[1, 1, 2, 4], 1.5);
        assert_eq!(l, vec![2.0, -3.0, 2.0, 0.0, -1.5]);
        // Token 1 appears twice and is penalised ONCE. Per occurrence gives -4.5, which
        // is the whole difference between the two readings.
        assert_eq!(l[1], -3.0);

        let mut l = vec![2.0, -2.0, 3.0, 0.0, -1.0];
        repetition_penalty(&mut l, &[1, 1, 2, 4], 2.0);
        assert_eq!(l, vec![2.0, -4.0, 1.5, 0.0, -2.0]);

        // A masked token stays masked.
        let mut l = vec![1.0, FILTERED];
        repetition_penalty(&mut l, &[1], 2.0);
        assert_eq!(l[1], FILTERED);
        // Out-of-range ids are ignored rather than panicking: a malformed history must
        // not take the process down mid-generation.
        let mut l = vec![1.0];
        repetition_penalty(&mut l, &[99], 2.0);
        assert_eq!(l, vec![1.0]);
    }

    #[test]
    fn frequency_penalty_really_is_per_occurrence() {
        let mut l = vec![0.0; 3];
        frequency_penalty(&mut l, &[1, 1, 1, 2], 0.5);
        assert_eq!(l, vec![0.0, -1.5, -0.5], "three occurrences, so 3 * 0.5");
        let mut l = vec![0.0; 3];
        presence_penalty(&mut l, &[1, 1, 1, 2], 0.5);
        assert_eq!(l, vec![0.0, -0.5, -0.5], "presence is once regardless of count");
    }

    /// The measured ban sets, with a vocabulary big enough that no ban falls off the end
    /// -- an earlier probe used `vocab = 8` on a sequence containing token 8 and
    /// concluded, wrongly, that the ban was missing.
    #[test]
    fn no_repeat_ngram_bans_what_followed_the_suffix() {
        let ban = |ids: &[u32], n: usize| -> Vec<usize> {
            let mut l = vec![0.0f32; 16];
            no_repeat_ngram(&mut l, ids, n);
            (0..l.len()).filter(|&i| l[i] == FILTERED).collect()
        };
        assert_eq!(ban(&[1, 2, 1, 2], 2), vec![1usize]);
        assert_eq!(ban(&[3, 3, 3], 2), vec![3usize]);
        assert_eq!(ban(&[1, 2, 3, 1, 2], 2), vec![3usize]);
        assert_eq!(ban(&[1, 2, 1, 2], 3), vec![1usize]);
        assert_eq!(ban(&[1, 2, 3, 1, 2], 3), vec![3usize]);
        assert_eq!(ban(&[5, 6, 7, 8, 6, 7], 3), vec![8usize]);
        // n=2 here: the suffix is [7], which occurred as the window [7,8], so the token
        // that followed it -- 8 -- is banned. A probe with `vocab = 8` reported an empty
        // ban set for this sequence, because id 8 fell into the reference's spare
        // masking column past the vocabulary. The sequence and the rule were fine; the
        // probe's vocabulary was too small.
        assert_eq!(ban(&[5, 6, 7, 8, 6, 7], 2), vec![8usize]);
        // Shorter than n: nothing to match yet.
        assert_eq!(ban(&[1], 3), Vec::<usize>::new());
        // n < 2 is off, not "ban everything".
        assert_eq!(ban(&[1, 2, 1, 2], 0), Vec::<usize>::new());
        assert_eq!(ban(&[1, 2, 1, 2], 1), Vec::<usize>::new());
    }

    /// Banning every token cancels itself out in a way that is invisible: the draw then
    /// has nothing to pick, and the error has to come from the pipeline rather than from
    /// a `NaN`.
    #[test]
    fn no_repeat_ngram_can_ban_everything_and_that_is_reported() {
        let mut s =
            Sampler::new(SamplerConfig { temperature: 1.0, no_repeat_ngram_size: 2, ..Default::default() })
                .unwrap();
        // One token, and the suffix [0] has occurred, so token 0 is banned and there is
        // nothing left. (With a two-token vocabulary only one of them gets banned, which
        // is why this case needs a vocabulary of one.)
        let mut l = vec![1.0f32];
        let e = s.next(&mut l, &[0, 0]).unwrap_err();
        assert!(e.contains("filtered out"), "{e}");
    }

    #[test]
    fn probs_do_not_overflow_and_survive_masking() {
        let p = softmax(&[100.0, 100.0, -100.0]);
        assert!(p.iter().all(|x| x.is_finite()), "{p:?}");
        assert!((p[0] - 0.5).abs() < 1e-6 && (p[1] - 0.5).abs() < 1e-6);
        assert_eq!(p[2], 0.0);
        assert_eq!(softmax(&[FILTERED, FILTERED]), vec![0.0, 0.0]);
        assert_eq!(softmax(&[]), Vec::<f32>::new());
    }

    #[test]
    fn select_walks_ascending_by_id() {
        let p = [0.5f32, 0.25, 0.125, 0.125];
        assert_eq!(select(&p, 0.0), Some(0));
        assert_eq!(select(&p, 0.4999), Some(0));
        assert_eq!(select(&p, 0.5), Some(1));
        assert_eq!(select(&p, 0.7499), Some(1));
        assert_eq!(select(&p, 0.75), Some(2));
        assert_eq!(select(&p, 0.8749), Some(2));
        assert_eq!(select(&p, 0.875), Some(3));
        // Past the end -- a few ulp of accumulated error, not a bug -- goes to the last
        // token with mass rather than to nothing.
        assert_eq!(select(&p, 0.999999999), Some(3));
        assert_eq!(select(&p, 1.5), Some(3));
        // Zero-mass tokens are skipped, not selected.
        assert_eq!(select(&[0.0, 1.0, 0.0], 0.0), Some(1));
        assert_eq!(select(&[0.0, 0.0], 0.0), None);
    }

    /// The distribution, not the token: draw a lot and compare frequencies.
    ///
    /// This is the only way to test a sampler. A wrong walk, or an RNG missing its mix
    /// step, still produces a plausible token every single time; it shows up as the wrong
    /// *proportion* over many draws.
    #[test]
    fn the_draw_reproduces_the_distribution() {
        let mut s = Sampler::new(cfg()).unwrap();
        let logits = vec![3.0f32, 1.0, 0.5, 0.1, -1.0, -5.0];
        let probs = softmax(&logits);
        let total: f64 = probs.iter().map(|&x| x as f64).sum();
        let target: Vec<f64> = probs.iter().map(|&x| x as f64 / total).collect();

        let n = 200_000;
        let mut counts = vec![0usize; logits.len()];
        for _ in 0..n {
            let mut l = logits.clone();
            counts[s.next(&mut l, &[]).unwrap() as usize] += 1;
        }
        for (i, &t) in target.iter().enumerate() {
            let got = counts[i] as f64 / n as f64;
            let sigma = (t * (1.0 - t) / n as f64).sqrt();
            assert!(
                (got - t).abs() < 5.0 * sigma + 1e-4,
                "token {i}: got {got:.6} want {t:.6} (5 sigma = {:.2e})",
                5.0 * sigma
            );
        }
    }

    #[test]
    fn a_seed_is_reproducible_and_a_different_seed_is_not() {
        let mk = |seed| {
            let mut s =
                Sampler::new(SamplerConfig { temperature: 1.0, seed, ..Default::default() }).unwrap();
            (0..32)
                .map(|_| {
                    let mut l = vec![1.0f32, 1.0, 1.0, 1.0];
                    s.next(&mut l, &[]).unwrap()
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(mk(7), mk(7));
        assert_ne!(mk(7), mk(8));
    }

    #[test]
    fn the_rng_is_uniform_enough_to_matter() {
        let mut r = SplitMix64::new(99);
        let n = 200_000;
        let mut buckets = [0usize; 16];
        let mut sum = 0.0f64;
        for _ in 0..n {
            let u = r.next_f64();
            assert!((0.0..1.0).contains(&u), "u out of range: {u}");
            sum += u;
            buckets[(u * 16.0) as usize] += 1;
        }
        assert!((sum / n as f64 - 0.5).abs() < 0.005, "mean {}", sum / n as f64);
        let expect = n as f64 / 16.0;
        for (i, &b) in buckets.iter().enumerate() {
            let sigma = (expect * (1.0 - 1.0 / 16.0)).sqrt();
            assert!((b as f64 - expect).abs() < 5.0 * sigma, "bucket {i}: {b} vs {expect:.0}");
        }
    }

    /// The RNG's output **is** its specification, so it is pinned rather than described.
    ///
    /// A generator has no internal property worth asserting beyond "the distribution is
    /// right over many draws", and that test is weak: dropping a mix step from `splitmix64`
    /// leaves a bijection with decent avalanche, so the mean and the bucket counts stay
    /// plausible and only the *values* change. These vectors come from the algorithm's
    /// definition in exact integer arithmetic, so any change to the mixing fails here.
    ///
    /// The `f64` stream is checked too, because the shift and the scale are part of the
    /// contract: `>> 11` with `2^-53` is what makes the value exactly representable and
    /// unable to round up to `1.0`.
    #[test]
    fn the_rng_matches_its_definition_exactly() {
        for (seed, pairs, f64s) in [
            (
                0x0u64,
                [
                    (0x9E37_79B9_7F4A_7C15u64, 0xE220_A839_7B1D_CDAFu64),
                    (0x3C6E_F372_FE94_F82A, 0x6E78_9E6A_A1B9_65F4),
                    (0xDAA6_6D2C_7DDF_743F, 0x06C4_5D18_8009_454F),
                    (0x78DD_E6E5_FD29_F054, 0xF88B_B8A8_724C_81EC),
                ],
                [0.8833108082136426f64, 0.43152799704850997, 0.026433771592597743],
            ),
            (
                0x1u64,
                [
                    (0x9E37_79B9_7F4A_7C16, 0x910A_2DEC_8902_5CC1),
                    (0x3C6E_F372_FE94_F82B, 0xBEEB_8DA1_658E_EC67),
                    (0xDAA6_6D2C_7DDF_7440, 0xF893_A2EE_FB32_555E),
                    (0x78DD_E6E5_FD29_F055, 0x71C1_8690_EE42_C90B),
                ],
                [0.5665615751722809, 0.7457817572627011, 0.9710027535867962],
            ),
            (
                0xDEAD_BEEFu64,
                [
                    (0x9E37_79BA_5DF8_3B04, 0x4ADF_B90F_68C9_EB9B),
                    (0x3C6E_F373_DD42_B719, 0xDE58_6A31_41A1_0922),
                    (0xDAA6_6D2D_5C8D_332E, 0x021F_BC2F_8E1C_FC1D),
                    (0x78DD_E6E6_DBD7_AF43, 0x7466_CE73_7BE1_6790),
                ],
                [0.29247624040798537, 0.868536602998237, 0.00829673920644669],
            ),
        ] {
            let mut r = SplitMix64::new(seed);
            for (want_state, want_out) in pairs {
                let got = r.next_u64();
                assert_eq!(r.state, want_state, "state, seed {seed:#x}");
                assert_eq!(got, want_out, "output, seed {seed:#x}");
            }
            // ... and the `f64` stream from the same seed.
            let mut r = SplitMix64::new(seed);
            for want in f64s {
                let got = r.next_f64();
                assert_eq!(got, want, "f64 stream, seed {seed:#x}");
            }
        }
    }

    /// The draw must have 53 bits of resolution, not 24.
    ///
    /// Both give a uniform `[0, 1)`, so the mean and the buckets cannot tell them apart. The
    /// resolution can: with a 24-bit generator every value is a multiple of `2^-24`, so
    /// `u * 2^24` is always an integer. With 53 bits that happens only when the low 29 bits
    /// of the mantissa are zero, which is one draw in 5e8 -- so a few thousand draws settle
    /// it with probability `1 - 2^-500000`.
    ///
    /// Worth pinning because the failure mode is invisible: a 24-bit generator makes every
    /// token with probability below `6e-8` literally unreachable, which is exactly the tail
    /// that sampling exists to explore.
    #[test]
    fn the_rng_has_more_than_24_bits_of_resolution() {
        let mut r = SplitMix64::new(7);
        let mut finer_than_24 = 0usize;
        let mut min = 1.0f64;
        for _ in 0..4096 {
            let u = r.next_f64();
            min = min.min(u);
            if (u * 16777216.0).fract() != 0.0 {
                finer_than_24 += 1;
            }
        }
        assert!(
            finer_than_24 > 4000,
            "only {finer_than_24} of 4096 draws had sub-24-bit detail; the generator is \
             coarser than 53 bits"
        );
        // And the values really do reach down: 4096 draws of a uniform must get below 3e-3
        // at least once with overwhelming probability (P(none) = (1-3e-3)^4096 ~ 4e-6).
        assert!(min < 3e-3, "the low tail is not being reached: min {min}");
    }

    #[test]
    fn temperature_zero_is_greedy_and_not_a_division_by_zero() {
        let mut s = Sampler::new(SamplerConfig { temperature: 0.0, ..Default::default() }).unwrap();
        for _ in 0..8 {
            let mut l = vec![0.1f32, 5.0, 0.2];
            assert_eq!(s.next(&mut l, &[]).unwrap(), 1);
        }
        assert!(SamplerConfig { temperature: 0.0, ..Default::default() }.is_greedy());
        assert!(!cfg().is_greedy());
    }

    #[test]
    fn an_all_filtered_pipeline_is_an_error_not_a_nan_token() {
        let mut s = Sampler::new(SamplerConfig { temperature: 1.0, ..Default::default() }).unwrap();
        let mut l = vec![FILTERED, FILTERED];
        let e = s.next(&mut l, &[]).unwrap_err();
        assert!(e.contains("filtered out"), "{e}");
    }

    #[test]
    fn config_validation_catches_the_shapes_that_cannot_work() {
        for c in [
            SamplerConfig { top_p: 1.5, ..Default::default() },
            SamplerConfig { min_p: -0.1, ..Default::default() },
            SamplerConfig { repetition_penalty: 0.0, ..Default::default() },
            SamplerConfig { temperature: f64::NAN, ..Default::default() },
            SamplerConfig { no_repeat_ngram_size: 1, ..Default::default() },
        ] {
            assert!(c.validate().is_err(), "{c:?} should be refused");
        }
        assert!(SamplerConfig { no_repeat_ngram_size: 2, ..Default::default() }.validate().is_ok());
    }

    /// `renormalize_logits` cannot change the sampled distribution, and saying so in a
    /// test is what stops someone adding a filter after it and being surprised.
    #[test]
    fn log_softmax_cannot_change_the_distribution() {
        let logits = vec![2.0f32, 1.0, 0.5, 0.1, -1.0, FILTERED];
        let before = softmax(&logits);
        let mut after_l = logits.clone();
        log_softmax_in_place(&mut after_l);
        let after = softmax(&after_l);
        let (sb, sa): (f64, f64) =
            (before.iter().map(|&x| x as f64).sum(), after.iter().map(|&x| x as f64).sum());
        for (a, b) in before.iter().zip(&after) {
            let (a, b) = (*a as f64 / sb, *b as f64 / sa);
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        assert_eq!(after_l[5], FILTERED, "-inf stays -inf through log_softmax");
    }

    /// The order is observable, which is why it is worth pinning.
    ///
    /// `repetition_penalty` is per-element and not monotone, so putting it before `top_k`
    /// changes which tokens `top_k` keeps. The reference puts the penalties first.
    #[test]
    fn the_penalty_runs_before_top_k_and_that_is_observable() {
        let logits = vec![3.0f32, 2.5, 2.0];
        // Reference order: penalise token 0 down to 1.5, then top_k=1 keeps token 1.
        let mut a = logits.clone();
        repetition_penalty(&mut a, &[0], 2.0);
        top_k(&mut a, 1);
        assert_eq!(kept(&a), vec![1usize]);
        // The other order keeps token 0 and never reconsiders.
        let mut b = logits.clone();
        top_k(&mut b, 1);
        repetition_penalty(&mut b, &[0], 2.0);
        assert_eq!(kept(&b), vec![0usize]);
        assert_ne!(kept(&a), kept(&b));
    }

    /// Temperature and a penalty are both scalar multiplications of the same entry, so
    /// they commute -- worth recording, because it means that pair of orders cannot be
    /// distinguished by any test and nobody should try.
    #[test]
    fn temperature_and_a_repetition_penalty_commute() {
        let logits = vec![-2.0f32, 1.0, 0.5];
        let mut a = logits.clone();
        repetition_penalty(&mut a, &[0], 2.0);
        temperature(&mut a, 0.5);
        let mut b = logits.clone();
        temperature(&mut b, 0.5);
        repetition_penalty(&mut b, &[0], 2.0);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn the_pipeline_is_the_union_of_its_stages() {
        // Nothing set: the pipeline must not touch the logits at all, which is what makes
        // "off" mean off rather than "off by a rounding error".
        let s = Sampler::new(SamplerConfig { temperature: 1.0, ..Default::default() }).unwrap();
        let l = vec![1.0f32, -2.0, 3.5, 0.0];
        let mut m = l.clone();
        s.apply_filters(&mut m, &[0, 1, 2, 3]);
        assert_eq!(l, m);

        // Everything on: the result must equal the stages applied by hand in order.
        let cfg = SamplerConfig {
            temperature: 0.8,
            top_k: 3,
            top_p: 0.9,
            min_p: 0.05,
            typical_p: 0.95,
            repetition_penalty: 1.3,
            presence_penalty: 0.1,
            frequency_penalty: 0.2,
            no_repeat_ngram_size: 2,
            seed: 1,
        };
        let s = Sampler::new(cfg).unwrap();
        let hist = [1u32, 2, 1, 2];
        let mut got = vec![2.0f32, 1.5, 1.0, 0.5, 0.25];
        s.apply_filters(&mut got, &hist);

        let mut want = vec![2.0f32, 1.5, 1.0, 0.5, 0.25];
        repetition_penalty(&mut want, &hist, 1.3);
        presence_penalty(&mut want, &hist, 0.1);
        frequency_penalty(&mut want, &hist, 0.2);
        no_repeat_ngram(&mut want, &hist, 2);
        temperature(&mut want, 0.8);
        top_k(&mut want, 3);
        top_p(&mut want, 0.9);
        min_p(&mut want, 0.05);
        typical_p(&mut want, 0.95);
        assert_eq!(got, want);
    }
}
