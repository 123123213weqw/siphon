#!/usr/bin/env python3
"""Build a chat-template conformance corpus from the reference implementation.

    make_chat_corpus.py <model-dir> <out.json>

The corpus records, for each case, the **inputs** and the **exact string the
reference produced** (or the exact error it raised). A Rust implementation then
has to reproduce those bytes, which is a much stronger statement than "the output
looks like a chat prompt".

Cases are chosen for the *structure* they exercise, one per branch of
`chat_template.jinja`, plus the branches that only fire on malformed input:

  * the four roles, and the unknown-role case in both of its positions -- before
    the last user query it fails the backwards scan, after it fails the loop, and
    the two produce different messages
  * `last_query_index`: an assistant before the last user query has no think
    block, an assistant after it does; a user message that is a wrapped
    `<tool_response>` does not end the backwards scan
  * `reasoning_content` as a string (which suppresses the inline `<think>`
    extraction) versus an inline `<think>` in the content
  * `add_generation_prompt` crossed with `enable_thinking`
  * the tool block: with and without a system message, with an empty system
    message, one tool and several, and argument values of every type -- because
    the template sends mappings and non-string sequences through `tojson` and
    everything else through `str`
  * the `tool` role: a single result, several in a row (which share one `user`
    block), and a result directly after a user message
  * `render_content`: plain strings, `None`, a list of text/image/video parts,
    `add_vision_id`, a system message that contains an image (refused), an item
    that is none of the three kinds, and a content that is not a string or a list
    at all
  * trimming, including `U+001C..U+001F` -- the four code points Python's
    `str.strip` removes and Rust's `str::trim` does not
  * CJK, emoji, combining marks, empty and whitespace-only content

The empty-message case is rendered through the template directly rather than
through `apply_chat_template`, which raises its own error first; the script
cross-checks the two paths on every other case so that the substitution is known
to be sound.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import jinja2
    from transformers import AutoTokenizer
except ImportError as e:  # pragma: no cover
    sys.exit(f"needs transformers and jinja2: {e}")


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
            "required": ["city"],
        },
    },
}

TOOL_WITH_AWKWARD_STRINGS = {
    "type": "function",
    "function": {
        "name": "f",
        "description": "a<b>c&d'e\"f\\g\x08\x1f",
        "parameters": {
            "properties": {
                "k": {"description": "中文 é 😀 \u2028"},
                "n": {"default": None},
                "t": {"default": True},
                "i": {"default": 42},
                "big": {"default": 1e16},
                # The float boundaries, which is where a hand-written `repr`
                # diverges: `1e15` and `1e-4` stay decimal, `1e17` and `1e-5`
                # do not, and `-0.0` and a 17-digit value each have their own
                # spelling.
                "at16": {"default": 1e15},
                "above16": {"default": 1e17},
                "at_1e_4": {"default": 1e-4},
                "below_1e_4": {"default": 1e-5},
                "neg_zero": {"default": -0.0},
                "zero": {"default": 0.0},
                "one": {"default": 1.0},
                "huge": {"default": 1e300},
                "seventeen": {"default": 0.3333333333333333},
                "lst": {"default": [1, 2, {"z": 0.1}]},
            }
        },
    },
}

TEXT_PART = {"type": "text", "text": "what is this?"}


def cases():
    """(name, inputs) pairs. `inputs` is what `apply_chat_template` is called with."""
    yield from _plain_cases()
    yield from _tool_block_cases()
    yield from _tool_role_cases()
    yield from _content_cases()
    yield from _error_cases()


def _plain_cases():
    yield ("single-turn", {"messages": [{"role": "user", "content": "Hi"}]})
    yield (
        "single-turn-no-generation-prompt",
        {"messages": [{"role": "user", "content": "Hi"}], "add_generation_prompt": False},
    )
    yield (
        "thinking-on",
        {
            "messages": [{"role": "user", "content": "Hi"}],
            "enable_thinking": True,
        },
    )
    yield (
        "system-first",
        {
            "messages": [
                {"role": "system", "content": "You are a terse assistant."},
                {"role": "user", "content": "Hi"},
            ]
        },
    )
    yield (
        "system-empty-and-whitespace",
        {
            "messages": [
                {"role": "system", "content": "   "},
                {"role": "user", "content": "Hi"},
            ]
        },
    )
    yield (
        "system-content-is-none",
        {"messages": [{"role": "system", "content": None}, {"role": "user", "content": "Hi"}]},
    )
    yield (
        "multi-turn-two",
        {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
                {"role": "assistant", "content": "d"},
            ],
            "add_generation_prompt": False,
        },
    )
    yield (
        "multi-turn-history-then-prompt",
        {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ]
        },
    )
    yield (
        "trailing-assistant",
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            "add_generation_prompt": False,
        },
    )
    # The four turns after a single query: every assistant is inside the turn.
    yield (
        "assistant-after-last-query",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ],
            "add_generation_prompt": True,
        },
    )
    yield (
        "reasoning-content-explicit",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "ans",
                    "reasoning_content": "the thought",
                },
            ],
            "add_generation_prompt": False,
        },
    )
    yield (
        "reasoning-content-empty-string-suppresses-extraction",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "before</think>\nafter",
                    "reasoning_content": "",
                },
            ],
            "add_generation_prompt": False,
        },
    )
    yield (
        "inline-think-in-content",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "<think>\nrz\n</think>\n\nans"},
            ],
            "add_generation_prompt": False,
        },
    )
    yield (
        "inline-think-no-closing-tag-is-left-alone",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a <think> b"},
            ],
            "add_generation_prompt": False,
        },
    )
    yield (
        "inline-think-repeated-takes-the-last",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "<think>\nA <think>\nB\n</think>\n\nC"},
            ],
            "add_generation_prompt": False,
        },
    )
    yield (
        "reasoning-content-in-history-is-not-think-wrapped",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a", "reasoning_content": "r"},
                {"role": "user", "content": "q2"},
            ],
            "add_generation_prompt": False,
        },
    )
    # Trimming: leading/trailing whitespace, and the four codepoints Python's
    # `str.strip` removes but Rust's `str::trim` does not.
    for name, s in [
        ("trim-spaces", "   Hi there  \n"),
        ("trim-tabs", "\t\tHi\t\t"),
        ("trim-fs", "\x1cHi\x1c"),
        ("trim-all-four-controls", "\x1c\x1d\x1e\x1f Hi \x1f\x1e\x1d\x1c"),
        ("trim-nel-nbsp", "\x85\xa0Hi\xa0\x85"),
        ("trim-ideographic-space", "\u3000Hi\u3000"),
        ("trim-zero-width-space-is-not-space", "\u200bHi\u200b"),
    ]:
        yield (name, {"messages": [{"role": "user", "content": s}]})
    for name, s in [
        ("unicode-cjk", "中国的首都是哪里？"),
        ("unicode-emoji", "héllo 😀🎉 world"),
        ("unicode-combining", "e\u0301 vs \u00e9"),
        ("unicode-rtl", "مرحبا שלום"),
        ("unicode-devanagari", "नमस्ते"),
        ("empty-content", ""),
        ("whitespace-only-content", "   \n\t  "),
        ("content-with-markers-inside", "a <|im_end|> b <think> c"),
        ("content-with-tool-response-markers-as-prose", "he said <tool_response> once"),
    ]:
        yield (name, {"messages": [{"role": "user", "content": s}]})
    yield (
        "content-none",
        {"messages": [{"role": "user", "content": None}]},
    )
    yield (
        "long-conversation",
        {
            "messages": sum(
                (
                    [
                        {"role": "user", "content": f"question {i}"},
                        {"role": "assistant", "content": f"answer {i}"},
                    ]
                    for i in range(4)
                ),
                [],
            )
            + [{"role": "user", "content": "final"}],
        },
    )


def _tool_block_cases():
    yield (
        "tools-no-system",
        {"messages": [{"role": "user", "content": "weather in Paris?"}], "tools": [WEATHER]},
    )
    yield (
        "tools-with-system",
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
            "tools": [WEATHER],
        },
    )
    yield (
        "tools-with-empty-system",
        {
            "messages": [
                {"role": "system", "content": "  "},
                {"role": "user", "content": "hi"},
            ],
            "tools": [WEATHER],
        },
    )
    yield (
        "tools-two",
        {"messages": [{"role": "user", "content": "hi"}], "tools": [WEATHER, WEATHER]},
    )
    # Every argument type, which decides `tojson` versus `str`.
    yield (
        "tools-awkward-strings-and-types",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [TOOL_WITH_AWKWARD_STRINGS],
        },
    )
    yield (
        "tools-empty-list-behaves-as-no-tools",
        {"messages": [{"role": "user", "content": "hi"}], "tools": []},
    )


def _tool_role_cases():
    yield (
        "tool-single",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "tool", "content": "R1"},
            ]
        },
    )
    yield (
        "tool-two-in-a-row-share-one-user-block",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "tool", "content": "R1"},
                {"role": "tool", "content": "R2"},
            ]
        },
    )
    yield (
        "tool-three-in-a-row",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "tool", "content": "R1"},
                {"role": "tool", "content": "R2"},
                {"role": "tool", "content": "R3"},
            ]
        },
    )
    yield (
        "tool-directly-after-user",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "tool", "content": "R"},
            ]
        },
    )
    yield (
        "tool-then-text-then-tool",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "tool", "content": "R1"},
                {"role": "user", "content": "R2"},
                {"role": "tool", "content": "R3"},
            ]
        },
    )
    # A whole multi-step tool loop: assistant calls, tool answers, assistant
    # calls again, each call after the last *unwrapped* user query.
    yield (
        "multi-step-tool-loop",
        {
            "messages": [
                {"role": "user", "content": "weather in Paris and London?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                        }
                    ],
                },
                {"role": "tool", "content": "18C"},
                {
                    "role": "assistant",
                    "content": "Paris is 18C.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"city": "London"}},
                        }
                    ],
                },
                {"role": "tool", "content": "12C"},
            ],
            "add_generation_prompt": True,
        },
    )
    yield (
        "tool-call-with-text-before-it",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "Let me look that up.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "f", "arguments": {"a": 1}},
                        }
                    ],
                },
            ],
            "add_generation_prompt": False,
        },
    )
    yield (
        "tool-call-without-function-wrapper",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "f", "arguments": {"a": 1}}],
                },
            ],
            "add_generation_prompt": False,
        },
    )
    yield (
        "tool-call-two-calls",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "both:",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "a", "arguments": {"x": "1"}}},
                        {"type": "function", "function": {"name": "b", "arguments": {}}},
                    ],
                },
            ],
            "add_generation_prompt": False,
        },
    )
    yield (
        "tool-call-argument-types",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "f",
                                "arguments": {
                                    "s": "str",
                                    "i": 42,
                                    "f": 1.5,
                                    "b": True,
                                    "bf": False,
                                    "n": None,
                                    "lst": [1, 2],
                                    "d": {"x": 1},
                                    "neg": -0.5,
                                    "tiny": 1e-5,
                                    "huge": 1e16,
                                    "q": "it's \"quoted\"",
                                },
                            },
                        }
                    ],
                },
            ],
            "add_generation_prompt": False,
        },
    )
    yield (
        "tool-call-no-arguments-key",
        {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": {"name": "f"}}],
                },
            ],
            "add_generation_prompt": False,
        },
    )


def _content_cases():
    yield (
        "content-text-part-only",
        {"messages": [{"role": "user", "content": [TEXT_PART]}]},
    )
    yield (
        "content-image-then-text",
        {
            "messages": [
                {"role": "user", "content": [{"type": "image"}, TEXT_PART]},
            ]
        },
    )
    yield (
        "content-two-images-with-ids",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "image"}, TEXT_PART],
                },
            ],
            "add_vision_id": True,
        },
    )
    yield (
        "content-image-url-form",
        {
            "messages": [
                {"role": "user", "content": [{"image_url": {"url": "x"}}, TEXT_PART]},
            ]
        },
    )
    yield (
        "content-video-then-text",
        {
            "messages": [
                {"role": "user", "content": [{"type": "video"}, TEXT_PART]},
            ]
        },
    )
    yield (
        "content-mixed-parts",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        TEXT_PART,
                        {"type": "image"},
                        {"type": "text", "text": " and this"},
                        {"type": "video"},
                    ],
                },
            ],
            "add_vision_id": True,
        },
    )
    yield (
        "content-parts-empty-text",
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": ""}]},
            ]
        },
    )
    # `{{ item.text }}` goes through Python's `str`, so a non-string text part is
    # not an error: it renders the value's string form.
    yield (
        "content-text-part-with-a-number",
        {
            "messages": [
                {"role": "user", "content": [{"text": 5}]},
            ]
        },
    )
    yield (
        "content-text-part-with-a-list",
        {
            "messages": [
                {"role": "user", "content": [{"text": ["a", 1]}]},
            ]
        },
    )
    yield (
        "content-text-part-with-none",
        {
            "messages": [
                {"role": "user", "content": [{"text": None}]},
            ]
        },
    )
    yield (
        "content-text-part-with-a-mapping",
        {
            "messages": [
                {"role": "user", "content": [{"text": {"a": 1}}]},
            ]
        },
    )
    yield (
        "content-text-part-with-a-bool",
        {
            "messages": [
                {"role": "user", "content": [{"text": False}]},
            ]
        },
    )
    yield (
        "content-system-list-of-text",
        {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "sys"}]},
                {"role": "user", "content": "q"},
            ]
        },
    )
    yield (
        "content-list-with-image-in-text-history",
        {
            "messages": [
                {"role": "user", "content": [{"type": "image"}, TEXT_PART]},
                {"role": "assistant", "content": "a cat"},
                {"role": "user", "content": "and now?"},
            ]
        },
    )


def _error_cases():
    """Cases where the template raises. The expected text is the exception's own."""
    yield ("error-empty-messages", {"messages": []})
    yield (
        "error-system-not-first",
        {"messages": [{"role": "user", "content": "q"}, {"role": "system", "content": "s"}]},
    )
    yield (
        "error-two-systems",
        {
            "messages": [
                {"role": "system", "content": "s1"},
                {"role": "system", "content": "s2"},
                {"role": "user", "content": "q"},
            ]
        },
    )
    yield (
        "error-no-user-query",
        {"messages": [{"role": "system", "content": "s"}]},
    )
    yield (
        "error-no-user-query-only-tool-responses",
        {"messages": [{"role": "user", "content": "<tool_response>\nR\n</tool_response>"}]},
    )
    yield (
        "error-unknown-role-only-message",
        {"messages": [{"role": "wizard", "content": "x"}]},
    )
    yield (
        "error-unknown-role-after-query",
        {"messages": [{"role": "user", "content": "q"}, {"role": "wizard", "content": "x"}]},
    )
    yield (
        "error-system-content-is-a-mapping",
        {"messages": [{"role": "system", "content": {"a": 1}}, {"role": "user", "content": "q"}]},
    )
    yield (
        "error-user-content-is-a-mapping",
        {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": {"a": 1}},
            ]
        },
    )
    yield (
        "error-user-content-is-a-number",
        {"messages": [{"role": "user", "content": 7}]},
    )
    yield (
        "error-system-contains-an-image",
        {
            "messages": [
                {"role": "system", "content": [{"type": "image"}]},
                {"role": "user", "content": "q"},
            ]
        },
    )
    yield (
        "error-system-contains-a-video",
        {
            "messages": [
                {"role": "system", "content": [{"type": "video"}]},
                {"role": "user", "content": "q"},
            ]
        },
    )
    yield (
        "error-item-type-in-content",
        {"messages": [{"role": "user", "content": [{"type": "audio"}]}]},
    )
    yield (
        "error-item-with-no-type-and-no-text",
        {"messages": [{"role": "user", "content": [{"nope": 1}]}]},
    )


def build_jinja_env(template: str):
    """The environment `transformers` compiles the template with.

    Rebuilt here so that the empty-message case can be rendered *through the
    template*: `apply_chat_template` intercepts that one before the template runs.
    `tojson` is the piece that matters -- `json.dumps` with
    `ensure_ascii=False`, and no HTML-safe escaping, which is why a tool
    description containing `<a>&'</a>` comes through unchanged.
    """
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False, default=str)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(
        jinja2.TemplateError(msg)
    )
    env.filters["raise_exception"] = env.globals["raise_exception"]
    return env.from_string(template)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    template = tok.chat_template
    if template is None:
        sys.exit(f"{model_dir} has no chat_template")
    # The template rendered directly is the authority, because it *is* the
    # template: `apply_chat_template` adds its own guards, and one of them (the
    # empty-message check) fires before the template ever runs. The two are
    # cross-checked on every case so the substitution is known to be sound.
    direct = build_jinja_env(template)

    out = []
    disagreements = 0
    wrappers = []
    for name, inp in cases():
        msgs = inp["messages"]
        kw = {k: v for k, v in inp.items() if k != "messages"}
        kw.setdefault("add_generation_prompt", True)
        tmpl_vars = {k: kw.pop(k) for k in ("enable_thinking", "add_vision_id") if k in kw}
        rec = {"name": name, "messages": msgs}
        # Template variables are recorded under their own names, which is what an
        # implementation reads.
        rec.update(tmpl_vars)
        rec.update(kw)

        # --- the template, directly ---
        alt, alt_error = None, None
        try:
            alt = direct.render(messages=msgs, **tmpl_vars, **kw)
        except Exception as e:  # noqa: BLE001
            alt_error = str(e).strip()
        # --- and through transformers ---
        ref, ref_error = None, None
        try:
            ref = tok.apply_chat_template(msgs, tokenize=False, **tmpl_vars, **kw)
        except Exception as e:  # noqa: BLE001
            ref_error = str(e).strip()

        if alt is not None and ref is not None and alt != ref:
            disagreements += 1
            print(f"  !! {name}: the two render paths disagree")
            if not args.quiet:
                print(f"     template: {alt!r}")
                print(f"     transformers: {ref!r}")
        elif alt is None and ref_error is None:
            disagreements += 1
            print(f"  !! {name}: the template raised but transformers did not: {alt_error!r}")
        elif alt is None and ref_error is not None and ref_error != alt_error:
            # transformers wraps or replaces some template errors. Recorded, not
            # treated as a failure, because the template's own message is what an
            # implementation of the template should produce.
            wrappers.append((name, alt_error, ref_error))

        if alt is not None:
            rec["expect_text"] = alt
            # The ids the reference tokenizer produces for that exact text. A
            # matching string with different ids is a real failure -- it is what a
            # missing added token looks like -- so both are recorded.
            ids = tok(alt, add_special_tokens=False)["input_ids"]
            via_template = tok.apply_chat_template(
                msgs, tokenize=True, return_dict=False, **tmpl_vars, **kw
            )
            if isinstance(via_template, dict):
                via_template = via_template["input_ids"]
            if list(via_template) != list(ids):
                disagreements += 1
                print(f"  !! {name}: encode(text) != apply_chat_template(tokenize=True)")
            rec["expect_ids"] = list(ids)
        else:
            rec["expect_error"] = alt_error
        out.append(rec)

    doc = {
        "mode": "chat_template.jinja (text + tools + tool role + vision + errors)",
        "generated_by": "tools/make_chat_corpus.py",
        "model_dir_name": model_dir.name,
        "cases": out,
    }
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    ok = sum(1 for r in out if "expect_text" in r)
    errs = sum(1 for r in out if "expect_error" in r)
    print(f"wrote {args.out}: {len(out)} cases ({ok} renders, {errs} errors)")
    if wrappers:
        print(f"  note: {len(wrappers)} case(s) where transformers reports its own error:")
        for name, mine, theirs in wrappers:
            print(f"    {name}: template={mine!r} transformers={theirs!r}")
    if disagreements:
        sys.exit(f"FAIL: {disagreements} disagreement(s)")


if __name__ == "__main__":
    main()
