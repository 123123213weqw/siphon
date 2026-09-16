#!/usr/bin/env python3
"""Probe the reference sampling pipeline: processor order and per-filter semantics.

Run on the reference box. Prints facts, writes nothing.

The point is that every one of these is a silent trap -- a wrong top-p boundary or
a symmetric repetition penalty produces *valid* probabilities and a *plausible*
token, so nothing about the output says it is wrong.
"""

import argparse
import copy
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# transformers 5.x merged the warpers into `logits_process`; `logits_warper` is gone.
from transformers.generation import logits_process as LP

MODEL = os.environ.get("QWEN35_MODEL_DIR", "")


def show(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def kinds(lst):
    return [type(x).__name__ for x in lst]


def P(v):
    """Print a small vector as prob/kept pairs."""
    return " ".join(f"{x:+.4f}" for x in v)


def main():
    ap = argparse.ArgumentParser(description="probe the reference's filter semantics")
    ap.add_argument("model_dir", nargs="?", default=MODEL, help="the checkpoint directory")
    ns = ap.parse_args()
    if not ns.model_dir:
        sys.exit("give a model directory, or set QWEN35_MODEL_DIR")
    model_dir = ns.model_dir
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    gc = model.generation_config
    print("transformers", __import__("transformers").__version__)
    print("default generation_config sampling fields:")
    for k in (
        "do_sample", "temperature", "top_k", "top_p", "min_p", "typical_p",
        "epsilon", "eta", "repetition_penalty", "presence_penalty",
        "frequency_penalty", "no_repeat_ngram_size", "renormalize_logits",
    ):
        print(f"   {k:24} {getattr(gc, k, 'ABSENT')!r}")

    show("1. warper class ORDER (what runs, in what order)")
    gc2 = copy.deepcopy(model.generation_config)
    gc2.do_sample = True
    gc2.temperature = 0.7
    gc2.top_k = 5
    gc2.top_p = 0.9
    gc2.min_p = 0.05
    gc2.typical_p = 0.9
    gc2.epsilon = 0.02
    gc2.eta = 0.3
    # In 5.x there is no `_get_logits_warper`; the warpers are built by
    # `_get_logits_processor` too, so this one call shows the whole order.
    print("warper order:", kinds(model._get_logits_processor(gc2)))
    gc3 = copy.deepcopy(model.generation_config)
    gc3.do_sample = True
    gc3.temperature = 0.7
    gc3.top_k = 5
    gc3.top_p = 0.9
    gc3.min_p = 0.05
    p = model._get_logits_processor(gc3)
    print("processor order (temperature/topk/top_p set):", kinds(p))
    gc4 = copy.deepcopy(model.generation_config)
    gc4.do_sample = True
    gc4.repetition_penalty = 1.2
    gc4.presence_penalty = 0.5
    gc4.frequency_penalty = 0.3
    gc4.no_repeat_ngram_size = 3
    gc4.min_new_tokens = 2
    gc4.min_length = 4
    gc4.forced_eos_token_id = None
    p4 = model._get_logits_processor(gc4)
    print("processor order (penalties set):", kinds(p4))
    gc5 = copy.deepcopy(model.generation_config)
    gc5.do_sample = True
    gc5.temperature = 0.7
    gc5.top_k = 5
    gc5.top_p = 0.9
    gc5.min_p = 0.05
    gc5.repetition_penalty = 1.2
    gc5.no_repeat_ngram_size = 3
    gc5.renormalize_logits = True
    print("FULL order (everything on):", kinds(model._get_logits_processor(gc5)))

    show("2. does each warper renormalise?")
    logits = torch.tensor([[3.0, 1.0, 0.5, 0.1, -1.0, -5.0]])
    print("raw softmax          ", P(torch.softmax(logits, -1)[0].tolist()))
    for name, cls, kw in [
        ("temperature 0.5", LP.TemperatureLogitsWarper, {"temperature": 0.5}),
        ("top_k 3", LP.TopKLogitsWarper, {"top_k": 3, "min_tokens_to_keep": 1}),
        ("top_p 0.9", LP.TopPLogitsWarper, {"top_p": 0.9, "min_tokens_to_keep": 1}),
        ("min_p 0.5", LP.MinPLogitsWarper, {"min_p": 0.5, "min_tokens_to_keep": 1}),
    ]:
        try:
            out = cls(**kw)(torch.zeros(1, 1, dtype=torch.long), logits.clone())
        except Exception as e:  # noqa: BLE001
            print(f"{name:20} RAISED {type(e).__name__}: {e}")
            continue
        sm = torch.softmax(out, -1)[0]
        print(f"{name:20} logits {P(out[0].tolist())}")
        print(f"{'':20} softmax {P(sm.tolist())}  sum={sm.sum():.6f}")

    show("3. top_p boundary rule (which token crosses the threshold)")
    # probs: .5 .25 .125 .0625 .0625 -- cumsum .5 .75 .875 .9375 1.0
    lg = torch.log(torch.tensor([[0.5, 0.25, 0.125, 0.0625, 0.0625]]))
    cum = torch.softmax(lg, -1).cumsum(-1)[0]
    print("probs  ", P(torch.softmax(lg, -1)[0].tolist()))
    print("cumsum ", P(cum.tolist()))
    for p in [1.0, 0.99, 0.95, 0.9, 0.875, 0.8, 0.76, 0.75, 0.6, 0.5, 0.49, 0.01, 0.0]:
        out = LP.TopPLogitsWarper(top_p=p, min_tokens_to_keep=1)(
            torch.zeros(1, 1, dtype=torch.long), lg.clone()
        )[0]
        kept = [i for i, v in enumerate(out.tolist()) if v != float("-inf")]
        print(f"  top_p={p:<6} keeps {kept}")

    show("4. min_p rule")
    for mp in [0.0, 0.01, 0.1, 0.25, 0.5, 0.99, 1.0]:
        out = LP.MinPLogitsWarper(min_p=mp, min_tokens_to_keep=1)(
            torch.zeros(1, 1, dtype=torch.long), lg.clone()
        )[0]
        kept = [i for i, v in enumerate(out.tolist()) if v != float("-inf")]
        print(f"  min_p={mp:<6} keeps {kept}")

    show("5. typical_p")
    for tp in [0.1, 0.5, 0.9, 0.99]:
        out = LP.TypicalLogitsWarper(mass=tp, min_tokens_to_keep=1)(
            torch.zeros(1, 1, dtype=torch.long), lg.clone()
        )[0]
        kept = [i for i, v in enumerate(out.tolist()) if v != float("-inf")]
        print(f"  typical_p={tp:<6} keeps {kept}")

    show("6. top_k with ties at the boundary")
    tie = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]])
    for k in [1, 2, 3, 4, 5, 6, 10]:
        out = LP.TopKLogitsWarper(top_k=k, min_tokens_to_keep=1)(
            torch.zeros(1, 1, dtype=torch.long), tie.clone()
        )[0]
        kept = [i for i, v in enumerate(out.tolist()) if v != float("-inf")]
        print(f"  top_k={k:<4} keeps {kept}")

    show("7. repetition / presence / frequency penalty formulas")
    ids = torch.tensor([[1, 1, 2, 4]])
    base = torch.tensor([[2.0, -2.0, 3.0, 0.0, -1.0]])
    print("input ids", ids[0].tolist(), " base logits", base[0].tolist())
    for rp in [1.0, 1.5, 2.0]:
        out = LP.RepetitionPenaltyLogitsProcessor(rp)(ids.clone(), base.clone())
        print(f"  repetition_penalty={rp:<5} -> {P(out[0].tolist())}")
    for name, cls in [
        ("presence_penalty", "PresencePenaltyLogitsProcessor"),
        ("frequency_penalty", "FrequencyPenaltyLogitsProcessor"),
    ]:
        if not hasattr(LP, cls):
            print(f"  {name}: NOT PRESENT in transformers {__import__('transformers').__version__}")
            continue
        for v in [0.0, 0.5, 2.0]:
            out = getattr(LP, cls)(v)(ids.clone(), base.clone())
            print(f"  {name}={v:<5} -> {P(out[0].tolist())}")

    show("8. no_repeat_ngram_size")
    for n in [1, 2, 3, 4]:
        seq = torch.tensor([[5, 6, 7, 8, 6, 7]])
        out = LP.NoRepeatNGramLogitsProcessor(n)(seq.clone(), torch.zeros(1, 8))
        banned = [i for i, v in enumerate(out[0].tolist()) if v == float("-inf")]
        print(f"  size={n} ids={seq[0].tolist()} bans {banned}")

    show("9. -inf / all -inf handling")
    with_inf = torch.tensor([[3.0, float("-inf"), 0.5, float("-inf")]])
    print("  softmax with -inf", P(torch.softmax(with_inf, -1)[0].tolist()))
    all_inf = torch.tensor([[float("-inf")] * 4])
    print("  softmax all -inf", torch.softmax(all_inf, -1)[0].tolist())
    print("  top_k 1 on the -inf vector:",
          LP.TopKLogitsWarper(1, 1)(torch.zeros(1, 1, dtype=torch.long),
                                               with_inf.clone())[0].tolist())

    show("10. how generate() draws from the probs")
    import inspect
    src = inspect.getsource(model._sample) if hasattr(model, "_sample") else ""
    print(src[:2500] if src else "(no _sample)")

    show("11. temperature=0")
    try:
        out = LP.TemperatureLogitsWarper(0.0)(
            torch.zeros(1, 1, dtype=torch.long), logits.clone()
        )
        print("  TemperatureLogitsWarper(0.0) ->", out[0].tolist())
    except Exception as e:  # noqa: BLE001
        print(f"  TemperatureLogitsWarper(0.0) RAISED {type(e).__name__}: {e}")

    show("12. end-to-end: does the model need the full processor list for this checkpoint")
    print("  eos_token_id:", tok.eos_token_id, " pad:", tok.pad_token_id)
    print("  gc.eos_token_id:", gc.eos_token_id)
    print("  tokenizer chat markers:", tok.convert_tokens_to_ids("<|im_end|>"))


if __name__ == "__main__":
    main()
