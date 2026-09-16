#!/usr/bin/env python3
"""Build a sampling-conformance corpus from the reference.

    make_sample_corpus.py <out.json> [--vocab N]

No model needed: the filters are functions of a logits vector, so the corpus is a set of
logits vectors and the reference's output for each. That makes it self-contained, fast to
regenerate, and exact -- the comparison is on the filtered **logits**, not on sampled
tokens, so nothing statistical is involved.

The input vectors are generated from a documented integer PRNG (`splitmix64`, the same
one the Rust sampler uses for its draw) so they are reproducible, but reproducibility is
not what makes the comparison exact: **the corpus stores the input logits**. The input is
whatever the file says, so nothing depends on the generator agreeing with anything. The
generator only has to avoid `f64` values that `f32` cannot hold, and the ones it builds
from integer arithmetic -- the majority -- are multiples of `2**-15`, which `f32`
represents exactly.

Almost every case is therefore compared **bit-exactly**: the filtered logits must match
the reference's, with `null` standing in for `-inf`. `temperature` is the exception,
because dividing by an arbitrary `t` is not exact in either language; those cases are
compared with a one-ulp tolerance.

The vectors are shaped to hit the boundaries rather than to look realistic:

  * `ties-at-the-boundary` -- the case where `top_k` keeps more than `k` and where
    `top_p`'s ascending scan picks a specific one of several equal tokens
  * `one-dominant` -- a `top_p` boundary that lands exactly on a cumulative sum
  * `flat` -- every token equal, so `top_k=1` keeps the whole vocabulary
  * `long-tail` -- one large logit and a decay, so `min_p` has something to cut
  * `already-masked` -- `-inf` present before the filters run
  * `extreme` -- `+100` and `-100`, which overflows an unguarded `exp`
  * `negative-only` -- every logit below zero, to pin the repetition penalty's branch

Cases are also included where the pipeline *order* is observable (`repetition_penalty`
before `top_k`), because a corpus that only ever sets one filter at a time cannot catch a
wrong order.
"""

import argparse
import json
import math
import sys

# Re-exported as a list so the Rust side and this file cannot drift: the checker reads
# `stage_order` and asserts its own pipeline matches.
STAGE_ORDER = [
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "no_repeat_ngram",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "typical_p",
]

VOCAB = 96


def make_logits(kind, n, seed):
    """The input vectors, from integer arithmetic only."""
    st = seed

    def nxt():
        nonlocal st
        st = (st + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = st
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 31
        # 24 bits mapped to (-128, 128) in steps of 2**-15, so every value is exact in f32.
        k = (z >> 40) & 0xFFFFFF
        return (k - (1 << 23)) / float(1 << 15)

    def rand():
        return [nxt() for _ in range(n)]

    def q(x):
        """A multiple of 2**-15 in (-128, 128), exact in f32."""
        return round(x * 32768) / 32768

    if kind == "ties-at-the-boundary":
        # Four equal maxima (so top_k must keep all four), then a long flat tail.
        return [q(4.0)] * 4 + [q(1.0)] * (n - 4)
    if kind == "one-dominant":
        # probs are 0.5, 0.25, 0.125, ... so a cumulative sum lands exactly on 0.75.
        return [q(math.log(0.5 / 2 ** i, math.e)) if i < 20 else q(-30.0) for i in range(n)]
    if kind == "flat":
        return [q(0.0)] * n
    if kind == "long-tail":
        return [q(6.0)] + [q(2.0 - 0.1 * i) for i in range(1, n)]
    if kind == "already-masked":
        v = rand()
        for i in range(0, n, 3):
            v[i] = float("-inf")
        return v
    if kind == "extreme":
        return [100.0, -100.0] + [q(v) for v in rand()[2:]]
    if kind == "negative-only":
        return [q(-0.5 - abs(x) * 0.01) for x in rand()]
    if kind == "two-equal-then-tail":
        return [q(3.0), q(3.0)] + [q(-2.0 - 0.05 * i) for i in range(2, n)]
    if kind == "one-hot":
        v = [q(-12.0)] * n
        v[7] = q(9.0)
        return v
    if kind == "random":
        return rand()
    raise ValueError(kind)


KINDS = [
    "ties-at-the-boundary",
    "one-dominant",
    "flat",
    "long-tail",
    "already-masked",
    "extreme",
    "negative-only",
    "two-equal-then-tail",
    "one-hot",
    "random",
]

# Each settings dict sets ONE filter family at a time, plus a few combinations. Setting
# them one at a time is what makes a failure name the filter.
SETTINGS = [
    {"temperature": 1.0},
    {"temperature": 0.7},
    {"temperature": 0.1},
    {"temperature": 2.0},
    {"top_k": 1},
    {"top_k": 3},
    {"top_k": 10},
    {"top_k": 96},
    {"top_p": 1.0},
    {"top_p": 0.99},
    {"top_p": 0.95},
    {"top_p": 0.9},
    {"top_p": 0.75},
    {"top_p": 0.5},
    {"top_p": 0.0},
    {"min_p": 0.05},
    {"min_p": 0.25},
    {"min_p": 0.5},
    {"min_p": 0.99},
    {"min_p": 1.0},
    {"typical_p": 0.1},
    {"typical_p": 0.5},
    {"typical_p": 0.9},
    {"typical_p": 0.99},
    {"repetition_penalty": 1.5},
    {"repetition_penalty": 2.0},
    {"repetition_penalty": 0.5},
    {"no_repeat_ngram_size": 2},
    {"no_repeat_ngram_size": 3},
    {"no_repeat_ngram_size": 4},
    # combinations, where the ORDER becomes observable
    {"repetition_penalty": 2.0, "top_k": 5},
    {"top_k": 5, "temperature": 0.5},
    {"top_k": 10, "top_p": 0.9},
    {"top_p": 0.9, "min_p": 0.05},
    {"min_p": 0.25, "typical_p": 0.9},
    {"temperature": 0.5, "typical_p": 0.5},
    {"repetition_penalty": 1.5, "no_repeat_ngram_size": 3, "top_p": 0.8},
    {"temperature": 0.8, "top_k": 20, "top_p": 0.95, "min_p": 0.02, "typical_p": 0.95},
    {"repetition_penalty": 1.2, "temperature": 0.6, "top_k": 40, "top_p": 0.9},
]

HISTORIES = {
    "empty": [],
    # A token repeated, to pin "once per distinct token" rather than per occurrence.
    "repeated": [5, 5, 5, 11, 5],
    "ngram2": [1, 2, 1, 2],
    "ngram3": [1, 2, 3, 1, 2],
    "ngram-repeat": [4, 4, 4],
    "with-it": [7, 7, 0, 7],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--vocab", type=int, default=VOCAB)
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers.generation import logits_process as LP

    vocab = args.vocab
    if vocab < 1:
        sys.exit("--vocab must be positive")
    for h in HISTORIES.values():
        for t in h:
            if t >= vocab:
                sys.exit(f"history token {t} is outside the vocabulary of {vocab}")

    cases = []
    for kind in KINDS:
        base = make_logits(kind, vocab, seed=0x1234567890ABCDEF)
        for setting in SETTINGS:
            # Histories only matter for the three filters that read them.
            needs_hist = any(
                k in setting
                for k in ("repetition_penalty", "no_repeat_ngram_size", "presence_penalty",
                          "frequency_penalty")
            )
            names = list(HISTORIES) if needs_hist else ["empty"]
            for hname in names:
                hist = HISTORIES[hname]
                if setting.get("no_repeat_ngram_size") and len(hist) < 1:
                    continue
                cases.append((kind, setting, hname, hist, base))

    out = []
    skipped = []
    for kind, setting, hname, hist, base in cases:
        lp = LP.LogitsProcessorList()
        # Built in the reference's own order, from the same fields `generate` reads, so
        # the corpus cannot disagree with the pipeline about the order.
        if setting.get("repetition_penalty"):
            lp.append(LP.RepetitionPenaltyLogitsProcessor(setting["repetition_penalty"]))
        if setting.get("no_repeat_ngram_size"):
            lp.append(LP.NoRepeatNGramLogitsProcessor(setting["no_repeat_ngram_size"]))
        if setting.get("temperature", 1.0) != 1.0:
            lp.append(LP.TemperatureLogitsWarper(setting["temperature"]))
        if setting.get("top_k"):
            lp.append(LP.TopKLogitsWarper(setting["top_k"], min_tokens_to_keep=1))
        if setting.get("top_p", 1.0) < 1.0:
            lp.append(LP.TopPLogitsWarper(setting["top_p"], min_tokens_to_keep=1))
        if setting.get("min_p", 0.0) > 0.0:
            lp.append(LP.MinPLogitsWarper(setting["min_p"], min_tokens_to_keep=1))
        if setting.get("typical_p", 1.0) < 1.0:
            lp.append(LP.TypicalLogitsWarper(setting["typical_p"], min_tokens_to_keep=1))

        logits = torch.tensor([base], dtype=torch.float32)
        # A genuinely empty `input_ids`, not a placeholder. Passing `[[0]]` for "no
        # history" makes the reference penalise token 0 as though it had appeared, and the
        # corpus then records a penalty for a history the case does not have -- which is
        # exactly the sort of thing that looks like an implementation bug.
        ids = torch.tensor([hist], dtype=torch.long) if hist else torch.zeros(1, 0, dtype=torch.long)
        try:
            got = lp(ids, logits.clone())[0].tolist()
        except Exception as e:  # noqa: BLE001
            skipped.append((kind, setting, hname, f"{type(e).__name__}: {e}"))
            continue

        # The corpus stores the input and the *complete* filtered vector, with -inf as
        # `null`. Comparing the whole vector rather than the kept set is what makes a
        # wrong temperature or a wrong penalty magnitude visible instead of merely a
        # wrong support.
        # Plain numbers, at full `f64` precision. A shorter spelling is possible -- every
        # value here is an `f32`, so `-198.83941650390625` is eighteen digits for a number
        # that has nine -- but `json.dump` writes floats with `repr` and controlling that
        # needs either a hand-rolled serialiser or a quote-stripping pass over the output.
        # It buys about 13%, which is not worth either. Said here so the size is a decision
        # rather than an oversight.
        def enc(v):
            if v == float("-inf"):
                return None
            if v != v:
                return "nan"
            return float(np.float32(v))

        out.append(
            {
                "name": f"{kind}|{hname}|{json.dumps(setting, sort_keys=True)}",
                "logits": [enc(v) for v in base],
                "history": hist,
                "settings": setting,
                "expect": [enc(v) for v in got],
            }
        )

    doc = {
        "mode": "logits filters, compared as filtered logits",
        "generated_by": "tools/make_sample_corpus.py",
        "reference": f"transformers {__import__('transformers').__version__}",
        "vocab": vocab,
        "stage_order": STAGE_ORDER,
        "note": (
            "`null` is -inf (a masked token) and \"nan\" is NaN. Inputs are multiples of "
            "2**-15 so they are exact in f32; `temperature` is the only stage compared "
            "with a tolerance, because dividing by an arbitrary t is not exact."
        ),
        "cases": out,
    }
    with open(args.out, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")

    # A corpus where every case keeps every token would pass trivially. Report the
    # distribution of how much each case filters, so that is visible rather than assumed.
    masked = [sum(1 for v in c["expect"] if v is None) for c in out]
    per_setting = {}
    for c in out:
        key = json.dumps(c["settings"], sort_keys=True)
        per_setting.setdefault(key, []).append(sum(1 for v in c["expect"] if v is None))
    print(f"wrote {args.out}: {len(out)} cases, vocab {vocab}")
    print(f"  masked tokens: min {min(masked)}, max {max(masked)}, of {vocab}")
    trivial = [k for k, v in per_setting.items() if min(v) == 0 and max(v) == 0]
    print(f"  setting groups that never mask anything: {len(trivial)}/{len(per_setting)}")
    if skipped:
        print(f"  {len(skipped)} case(s) the reference raised on:")
        for kind, setting, hname, err in skipped[:5]:
            print(f"    {kind}|{hname}|{setting}: {err}")
    n_distinct = len({tuple(c["settings"].items()) for c in out})
    print(f"  distinct settings: {n_distinct}")
    import os
    print(f"  file size: {os.path.getsize(args.out) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
