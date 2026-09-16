#!/usr/bin/env python3
"""Does `samplecheck` have teeth?

    tools/mutate_sample.py --corpus sample_corpus.json [--only SUBSTRING]

Each mutation is a plausible mistake in the sampling filters, the draw, or the pipeline
order. Every one is applied on its own, the tree is rebuilt, and `samplecheck` is run against
the reference corpus.

This matters more here than anywhere else in the tree. Every filter in `sample.rs` produces
*valid probabilities and a plausible token* when it is wrong. A symmetric repetition penalty,
or a `top_k` that sorts instead of thresholding, does not crash, does not produce `NaN`, and
does not look wrong in any single sample -- it changes the distribution in a way that only an
exact comparison, or a lot of draws, can see.

Mutations are grouped by the *property* they break rather than the line they change:

  * `top_k` that sorts and truncates (the natural implementation, and wrong with ties)
  * `top_p` that scans descending (equivalent except at ties, which are common), that tests
    `<` instead of `<=`, that loses `min_tokens_to_keep`, or whose threshold comes from an
    `f32` config value
  * `min_p` against an unnormalised softmax, and with a non-strict comparison
  * `typical_p` with a plain `sum` instead of `nansum`, and with `last_ind` off by one
  * a repetition penalty applied per occurrence, and a symmetric one
  * `presence_penalty` and `frequency_penalty` swapped
  * `no_repeat_ngram` banning the window's first token
  * `temperature` after the penalty instead of before
  * a draw that walks descending, and one that takes the last token over the threshold
  * an RNG missing a mix step, and one with 24 bits of resolution

Three of these are order or RNG mutations that only the combination cases and the frequency
check can see, which is why the corpus has combination cases and why `samplecheck` has a
`--frequency` mode.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SAMPLE = "gdn/src/sample.rs"

# (file, regex, replacement, label)
MUTATIONS = [
    # --- top_k -------------------------------------------------------------
    (
        SAMPLE,
        r"    let mut vals = logits\.to_vec\(\);\n"
        r"    let \(_, kth, _\) = vals\.select_nth_unstable_by\(k - 1, \|a, b\| \{\n"
        r"        b\.partial_cmp\(a\)\.unwrap_or\(std::cmp::Ordering::Equal\)\n"
        r"    \}\);\n"
        r"    let kth = \*kth;\n"
        r"    for v in logits\.iter_mut\(\) \{\n"
        r"        if \*v < kth \{\n"
        r"            \*v = FILTERED;\n"
        r"        \}\n"
        r"    \}",
        "    let mut order: Vec<usize> = (0..logits.len()).collect();\n"
        "    order.sort_by(|&a, &b| {\n"
        "        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)\n"
        "    });\n"
        "    for &i in order.iter().skip(k) {\n"
        "        logits[i] = FILTERED;\n"
        "    }",
        "top_k sorts and truncates, so ties do not all survive",
    ),
    (
        SAMPLE,
        r"        if \*v < kth \{",
        "        if *v <= kth {",
        "top_k drops the token exactly at the boundary",
    ),
    # --- top_p -------------------------------------------------------------
    (
        SAMPLE,
        r"pub fn top_p_threshold\(p: f64\) -> f32 \{\n    \(1\.0 - p\) as f32\n\}",
        "pub fn top_p_threshold(p: f64) -> f32 {\n    (1.0 - p as f32 as f64) as f32\n}",
        "the top_p threshold comes from an f32 config value",
    ),
    (
        SAMPLE,
        r"        acc \+= probs\[i\];\n        if acc <= threshold \{",
        "        acc += probs[i];\n        if acc < threshold {",
        "top_p tests < instead of <=",
    ),
    (
        SAMPLE,
        r"    if logits\[best\] == FILTERED \{\n        logits\[best\] = best_val;\n    \}",
        "    let _ = best_val;",
        "top_p can mask everything (min_tokens_to_keep = 1 is lost)",
    ),
    (
        SAMPLE,
        r"    order\.sort_by\(\|&a, &b\| \{\n"
        r"        logits\[a\]\.partial_cmp\(&logits\[b\]\)\.unwrap_or\(std::cmp::Ordering::Equal\)\n"
        r"    \}\);",
        "    let mut order: Vec<usize> = order.into_iter().rev().collect();\n"
        "    order.sort_by(|&a, &b| {\n"
        "        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)\n"
        "    });",
        "top_p scans descending instead of ascending",
    ),
    # --- min_p -------------------------------------------------------------
    (
        SAMPLE,
        r"    let top = probs\.iter\(\)\.copied\(\)\.fold\(0\.0f32, f32::max\);\n"
        r"    let floor = min_p as f32 \* top;",
        "    let raw = probs.iter().copied().fold(0.0f32, f32::max);\n"
        "    let total: f32 = probs.iter().sum();\n"
        "    let floor = min_p as f32 * raw / total;",
        "min_p compares against an unnormalised softmax",
    ),
    (
        SAMPLE,
        r"        if q < floor \{",
        "        if q <= floor {",
        "min_p drops the token exactly at the threshold",
    ),
    # --- typical_p ---------------------------------------------------------
    (
        SAMPLE,
        r"        if is_positive\(q\) && logits\[i\] != FILTERED \{\n"
        r"            ent -= q\.ln\(\) \* q;\n"
        r"        \}",
        "        if logits[i] != FILTERED {\n"
        "            ent -= q.ln() * q;\n"
        "        }",
        "typical_p uses sum instead of nansum, so a masked token makes the entropy NaN",
    ),
    (
        SAMPLE,
        r"        if acc < mass as f32 \{\n            last_ind = k;\n        \}",
        "        if acc <= mass as f32 {\n            last_ind = k;\n        }",
        "typical_p's last_ind is off by one",
    ),
    (
        SAMPLE,
        r"    let cutoff = shifted\[order\[last_ind\]\];",
        "    let cutoff = shifted[order.last().copied().unwrap_or(0)];",
        "typical_p never cuts anything",
    ),
    # --- the penalties -----------------------------------------------------
    (
        SAMPLE,
        r"    for i in distinct_tokens\(history, logits\.len\(\)\) \{\n"
        r"        if logits\[i\] == FILTERED \{\n"
        r"            continue;\n"
        r"        \}\n"
        r"        let s = logits\[i\];\n"
        r"        logits\[i\] = if s < 0\.0 \{ s \* penalty \} else \{ s / penalty \};\n"
        r"    \}",
        "    for &t in history {\n"
        "        let i = t as usize;\n"
        "        if i >= logits.len() || logits[i] == FILTERED {\n"
        "            continue;\n"
        "        }\n"
        "        let s = logits[i];\n"
        "        logits[i] = if s < 0.0 { s * penalty } else { s / penalty };\n"
        "    }",
        "the repetition penalty is applied once per occurrence, not per distinct token",
    ),
    (
        SAMPLE,
        r"        logits\[i\] = if s < 0\.0 \{ s \* penalty \} else \{ s / penalty \};",
        "        logits[i] = s / penalty;",
        "the repetition penalty is symmetric (no sign branch)",
    ),
    (
        SAMPLE,
        r"    for i in distinct_tokens\(history, logits\.len\(\)\) \{\n"
        r"        if logits\[i\] != FILTERED \{\n"
        r"            logits\[i\] -= penalty;\n"
        r"        \}\n"
        r"    \}",
        "    for &t in history {\n"
        "        let i = t as usize;\n"
        "        if i < logits.len() && logits[i] != FILTERED {\n"
        "            logits[i] -= penalty;\n"
        "        }\n"
        "    }",
        "presence_penalty becomes per-occurrence (i.e. a frequency penalty)",
    ),
    (
        SAMPLE,
        r"            Some\(\(_, c\)\) => \*c \+= 1,",
        "            Some((_, c)) => *c += 0,",
        "frequency_penalty becomes per-token (i.e. a presence penalty)",
    ),
    # --- no_repeat_ngram ---------------------------------------------------
    (
        SAMPLE,
        r"        if &history\[w\.\.w \+ n - 1\] == prefix && !banned\.contains\(&history\[w \+ n - 1\]\) \{\n"
        r"            banned\.push\(history\[w \+ n - 1\]\);\n"
        r"        \}",
        "        if &history[w..w + n - 1] == prefix && !banned.contains(&history[w]) {\n"
        "            banned.push(history[w]);\n"
        "        }",
        "no_repeat_ngram bans the window's first token instead of its last",
    ),
    (
        SAMPLE,
        r"        if &history\[w\.\.w \+ n - 1\] == prefix && !banned\.contains\(&history\[w \+ n - 1\]\) \{",
        "        if &history[w + 1..w + n] == prefix {",
        "no_repeat_ngram matches the wrong window",
    ),
    (
        SAMPLE,
        r"    if n < 2 \|\| history\.len\(\) < n \{",
        "    if n < 1 || history.len() < n {",
        "no_repeat_ngram treats size 1 as on, banning every repeat",
    ),
    # --- the pipeline order ------------------------------------------------
    (
        SAMPLE,
        r"        if c\.temperature > 0\.0 && c\.temperature != 1\.0 \{\n"
        r"            temperature\(logits, c\.temperature as f32\);\n"
        r"        \}\n"
        r"        if c\.top_k >= 1 \{",
        "        if c.top_k >= 1 {",
        "temperature runs after top_k instead of before it",
    ),
    # --- the draw ----------------------------------------------------------
    (
        SAMPLE,
        r"    for \(i, &p\) in probs\.iter\(\)\.enumerate\(\) \{",
        "    for (i, &p) in probs.iter().enumerate().rev() {",
        "the draw walks descending by id",
    ),
    (
        SAMPLE,
        r"        if u < acc \{\n            return Some\(i\);\n        \}",
        "        if u < acc {\n            last = Some(i);\n        }",
        "the draw takes the last token over the threshold instead of the first",
    ),
    (
        SAMPLE,
        r"        acc \+= p as f64;\n        if u < acc \{",
        "        acc += (p as f64) / 2.0;\n        if u < acc {",
        "the draw halves each mass, so the walk is biased toward the low ids",
    ),
    # --- the RNG -----------------------------------------------------------
    (
        SAMPLE,
        r"        z = \(z \^ \(z >> 27\)\)\.wrapping_mul\(0x94D0_49BB_1331_11EB\);\n"
        r"        z \^ \(z >> 31\)",
        "        z ^ (z >> 31)",
        "the RNG drops its second mix step",
    ),
    (
        SAMPLE,
        r"        \(self\.next_u64\(\) >> 11\) as f64 \* \(1\.0 / 9007199254740992\.0\)",
        "        (self.next_u64() >> 40) as f64 * (1.0 / 16777216.0)",
        "the RNG has 24 bits of resolution instead of 53",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="sample_corpus.json")
    ap.add_argument("--binary", default="./target/release/samplecheck")
    ap.add_argument("--frequency", type=int, default=0, help="draws per case, for the RNG ones")
    ap.add_argument("--only", default=None, help="run only labels containing this")
    args = ap.parse_args()

    if not Path(args.corpus).exists():
        sys.exit(f"no corpus at {args.corpus}")

    tmp = Path(tempfile.mkdtemp())
    # One backup per file, under its own name: backing two files to one name makes every
    # injection look like a compile failure, which is a bug an earlier harness had.
    shutil.copy(SAMPLE, tmp / "sample.rs")

    def restore():
        shutil.copy(tmp / "sample.rs", SAMPLE)

    caught = missed = untested = 0
    try:
        print("############ mutations ############")
        for path, pattern, replacement, label in MUTATIONS:
            if args.only and args.only not in label:
                continue
            restore()
            src = Path(path).read_text()
            patched, count = re.subn(pattern, lambda _m: replacement, src)
            if count != 1:
                print(f"  [pattern matched {count}x, not 1]  {label}")
                untested += 1
                continue
            Path(path).write_text(patched)

            build = subprocess.run(["cargo", "build", "--release"], capture_output=True, text=True)
            if build.returncode != 0:
                print(f"  [does not compile]     {label}")
                untested += 1
                continue

            # Two oracles, because some of these properties are only visible to one of them.
            # The corpus pins the filters against the reference, which is the strongest
            # check available for anything the reference implements. The unit tests cover
            # what it does not: `presence_penalty` and `frequency_penalty` are absent from
            # this transformers version, the `select` walk and the RNG have no reference to
            # compare against, and a boundary condition can happen to be invisible in the
            # corpus's particular logits. A mutation counts as caught if either fails.
            unit = subprocess.run(
                ["cargo", "test", "--release", "--lib"], capture_output=True, text=True
            )
            unit_failed = unit.returncode != 0
            cmd = [args.binary, args.corpus]
            if args.frequency:
                cmd += ["--frequency", str(args.frequency)]
            run = subprocess.run(cmd, capture_output=True, text=True)
            corpus_failed = "RESULT: PASS" not in run.stdout

            if not unit_failed and not corpus_failed:
                print(f"  [MISSED]               {label}")
                missed += 1
                continue
            lines = [l.strip() for l in run.stdout.splitlines()]
            detail = next((l for l in lines if l.startswith("!!")), "")
            if not detail:
                detail = next((l for l in lines if "logits" in l and "bad" in l), "")
            if not detail:
                detail = next((l for l in lines if "sigma" in l), "")
            if not detail:
                detail = next((l for l in lines if "target(s)" in l), "")
            if not corpus_failed:
                # The unit tests caught it; name the test.
                detail = next(
                    (
                        l.strip()
                        for l in unit.stdout.splitlines()
                        if l.strip().startswith("test ") and "FAILED" in l
                    ),
                    "the unit tests failed",
                )
            which = []
            if corpus_failed:
                which.append("corpus")
            if unit_failed:
                which.append("unit tests")
            print(f"  [caught by {' + '.join(which)}] {label}")
            if detail:
                print(f"           {detail}")
            caught += 1
    finally:
        restore()
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(["cargo", "build", "--release"], capture_output=True)

    print()
    print("############ summary ############")
    print(f"  caught {caught}   missed {missed}   not-tested {untested}")
    if missed or untested:
        print("  RESULT: FAIL")
        return 1
    print("  RESULT: PASS (every injection was caught)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
