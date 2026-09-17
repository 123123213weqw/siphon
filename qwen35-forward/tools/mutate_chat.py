#!/usr/bin/env python3
"""Does `chatcheck` have teeth?

    tools/mutate_chat.py [--corpus FILE] [--model-dir DIR]

Every mutation below is a plausible mistake in the chat template or in the
Python-compatible JSON writer: a wrong trim, the think block on the wrong
assistants, a missing end marker, `str` where `tojson` belongs, a float threshold
off by one. Each is applied on its own, the tree is rebuilt, and `chatcheck` is
run against the reference corpus. A mutation that is *not* caught means the
corpus is not testing what it claims to -- an assertion nobody has seen fail is
not evidence.

Three things this harness has to get right, all of which were bugs in a one-off
version of it:

* **Separate backups per file.** Backing two files up to the same name makes
  every injection look like a compile failure.
* **A compile failure is not a catch.** It is reported separately, because a
  mutation that does not build has not been tested at all.
* **Patterns are regular expressions, written with `\\x5c` and `\\x22`** for
  backslash and double quote. The Rust being mutated is full of `'\\'` and `"\""`,
  and a literal-match harness needs four levels of quoting to express them, which
  is where the first version of this file broke.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHAT = "gdn/src/chat.rs"
CHATPARSE = "gdn/src/chatparse.rs"
# `pyjson` moved to the shared `shell-text` crate; the mutation still has to find
# the real source, so this path reaches out of the tree on purpose.
JSON = "../shell/crates/shell-text/src/pyjson.rs"

BS = chr(0x5C)  # a backslash, for building replacements that need a literal one
Q = chr(0x22)  # a double quote


def esc(s: str) -> str:
    """A literal Rust snippet as a regex, so the quoting cannot go wrong."""
    return re.escape(s)


# `out.push_str("\": ");` and the three lines around the backslash arm of
# `json_escape_into`. Written by composition because the Rust is mostly quotes and
# backslashes; `py_repr_str` has the same backslash arm, so the match has to
# include its neighbour to be unique.
JSON_KEY_SEP_OLD = "out.push_str(" + Q + BS + Q + ": " + Q + ");"
JSON_KEY_SEP_NEW = "out.push_str(" + Q + BS + Q + ":" + Q + ");"

# (file, regex, replacement, label)
MUTATIONS = [
    # --- trimming ---------------------------------------------------------
    (
        CHAT,
        r"s\.trim_matches\(is_python_space\)\.to_string\(\)",
        "s.to_string()",
        "no trimming at all",
    ),
    (
        CHAT,
        r"s\.trim_matches\(is_python_space\)\.to_string\(\)",
        "s.trim().to_string()",
        "Rust's trim instead of Python's strip (U+001C..U+001F)",
    ),
    # --- last_query_index -------------------------------------------------
    (
        CHAT,
        r"if i > last_query_index \{",
        "if i <= last_query_index {",
        "the think block goes on the wrong assistants",
    ),
    (
        CHAT,
        r"for i in \(0\.\.n\)\.rev\(\) \{",
        "for i in 0..n {",
        "last_query_index takes the first user, not the last",
    ),
    (
        CHAT,
        r'if !\(c\.starts_with\("<tool_response>"\) && c\.ends_with\("</tool_response>"\)\) \{',
        "if true {",
        "a wrapped tool response ends the backwards scan",
    ),
    # --- the assistant block ---------------------------------------------
    (
        CHAT,
        r'out\.push_str\("\x5cn\x5cn<tool_call>\x5cn<function="\);',
        'out.push_str("<tool_call>\\n<function=");',
        "the blank line before a tool call is dropped",
    ),
    (
        CHAT,
        r"Value::Array\(_\) \| Value::Object\(_\) => pyjson::dumps\(v\),",
        "Value::Array(_) | Value::Object(_) => pyjson::py_str(v),",
        "tool arguments go through str instead of tojson",
    ),
    (
        CHAT,
        r"out\.push_str\(IM_END\);\n\s+out\.push\('\\n'\);\n\s+\}\n\s+// Consecutive",
        "out.push('\\n');\n            }\n            // Consecutive",
        "an assistant message is not closed with <|im_end|>",
    ),
    (
        CHAT,
        r"if !jinja_trim\(&content\)\.is_empty\(\) \{",
        "if true {",
        "the first tool call is always preceded by a blank line",
    ),
    # --- the generation prompt -------------------------------------------
    (
        CHAT,
        r"if req\.opts\.enable_thinking \{",
        "if !req.opts.enable_thinking {",
        "the thinking toggle is inverted",
    ),
    # --- the tool role ----------------------------------------------------
    (
        CHAT,
        r"if i > 0 && req\.messages\[i - 1\]\.role != Role::Tool \{",
        "if i > 0 {",
        "every tool result gets its own user block",
    ),
    # --- rendering --------------------------------------------------------
    (
        CHAT,
        r'out\.push_str\(&format!\("Picture \{\}: ", \*images\)\);',
        'out.push_str(&format!("Picture {}: ", 0));',
        "the vision id counter never advances",
    ),
    # --- the other half: parsing -----------------------------------------
    (
        CHATPARSE,
        r'let cut = body\.find\("<tool_call>"\)\.unwrap_or\(body\.len\(\)\);',
        "let cut = 0;",
        "the parser throws the content away when there is a tool call",
    ),
    (
        CHATPARSE,
        r'\.split_once\("\\n</parameter>\\n"\)',
        '.split_once("</parameter>")',
        "the parser accepts an unterminated <parameter>",
    ),
    # --- Python-compatible JSON ------------------------------------------
    (
        JSON,
        esc(JSON_KEY_SEP_OLD),
        JSON_KEY_SEP_NEW,
        "compact JSON key separator",
    ),
    (
        JSON,
        r'out\.push_str\(", "\);(?=\n\s+\}\n\s+dumps_into\(e, out\);)',
        "out.push(',');",
        "compact JSON array separator",
    ),
    (
        JSON,
        r"if decpt <= -4 \|\| decpt > 16 \{",
        "if decpt <= -4 || decpt > 15 {",
        "the float exponent threshold is off by one",
    ),
    (
        JSON,
        esc("            '" + Q + "' => out.push_str(" + Q + BS + BS + BS + Q + Q + "),"),
        "            '" + Q + "' => out.push('" + Q + "'),",
        'a double quote in a tool description is not escaped',
    ),
    (
        JSON,
        r"'\x5cu\{8\}' => out\.push_str\(\x22\x5c\x5cb\x22\),",
        r"'\u{8}' => out.push('\u{8}'),",
        "a backspace is written literally instead of as \\b",
    ),
]

FILES = {CHAT, CHATPARSE, JSON}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/tmp/chat_corpus.json")
    ap.add_argument("--model-dir", required=True, help="the checkpoint to test against")
    ap.add_argument("--binary", default="./target/release/chatcheck")
    ap.add_argument("--keep-going", action="store_true", help="report MISSED but exit 0")
    args = ap.parse_args()

    if not Path(args.corpus).exists():
        sys.exit(f"no corpus at {args.corpus}; run tools/make_chat_corpus.py first")

    tmp = Path(tempfile.mkdtemp())
    # One backup per file, under its own name.
    for f in sorted(FILES):
        shutil.copy(f, tmp / Path(f).name)

    def restore():
        for f in sorted(FILES):
            shutil.copy(tmp / Path(f).name, f)

    caught = missed = untested = 0
    try:
        print("############ mutations ############")
        for path, pattern, replacement, label in MUTATIONS:
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

            run = subprocess.run(
                [args.binary, args.model_dir, args.corpus], capture_output=True, text=True
            )
            if "RESULT: PASS" in run.stdout:
                print(f"  [MISSED]               {label}")
                missed += 1
                continue
            first = next(
                (l.strip() for l in run.stdout.splitlines() if l.strip().startswith("!!")),
                "(no detail)",
            )
            print(f"  [caught] {label}")
            print(f"           {first}")
            caught += 1
    finally:
        restore()
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(["cargo", "build", "--release"], capture_output=True)

    print()
    print("############ summary ############")
    print(f"  caught {caught}   missed {missed}   not-tested {untested}")
    if untested or (missed and not args.keep_going):
        print("  RESULT: FAIL")
        return 1
    print("  RESULT: PASS (every injection was caught)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
