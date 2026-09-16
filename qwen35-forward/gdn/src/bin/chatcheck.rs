//! `chatcheck` -- run the chat template against the committed conformance corpus.
//!
//! ```text
//! chatcheck <model-dir> <corpus.json> [--verbose] [--max N]
//! ```
//!
//! The corpus is produced by `tools/make_chat_corpus.py` from the reference: every
//! case carries the inputs and the **exact text** `chat_template.jinja` produced,
//! or the **exact error** it raised. So this compares bytes, not impressions.
//!
//! Three things are checked per case, and all three are needed:
//!
//! * the rendered **text** matches byte for byte, which is the main check;
//! * the **token ids** of that text match the reference's, which catches a
//!   mistake the text check cannot -- e.g. a marker that renders correctly but is
//!   not a single added token, so the model would see different input;
//! * the **error** matches, for the cases where the template refuses the input,
//!   because a template that accepts what the reference rejects has silently
//!   changed the prompt.
//!
//! Plus the identity that ties this tree's two halves together: render, parse,
//! render again, and the bytes must be unchanged. `chatparse` is checked that way
//! here as well as in its own tests.

use std::process::ExitCode;

use gdn::chat::{self, Request};
use gdn::chatparse;
use gdn::pyjson;
use gdn::tokenizer::Tokenizer;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let flag = |n: &str| args.iter().any(|a| a == n);
    let val = |n: &str| -> Option<String> {
        args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned()
    };
    let positional: Vec<&String> = args.iter().skip(1).filter(|a| !a.starts_with("--")).collect();
    if positional.len() < 2 {
        eprintln!("usage: chatcheck <model-dir> <corpus.json> [--verbose] [--max N]");
        return ExitCode::from(2);
    }
    let verbose = flag("--verbose");
    let max: Option<usize> = val("--max").and_then(|s| s.parse().ok());

    // The tokenizer is loaded because a case's ids are as much a part of the
    // expected output as its text.
    let (tk, tk_info) = match Tokenizer::from_model_dir(positional[0]) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let raw = match std::fs::read(positional[1]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {}: {e}", positional[1]);
            return ExitCode::FAILURE;
        }
    };
    let doc = match pyjson::parse(&String::from_utf8_lossy(&raw)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {}: {e}", positional[1]);
            return ExitCode::FAILURE;
        }
    };
    let mode = doc.get("mode").and_then(|m| m.as_str()).unwrap_or("?");
    let cases = match doc.get("cases").and_then(|c| c.as_array()) {
        Some(c) => c,
        None => {
            eprintln!("error: {}: no `cases`", positional[1]);
            return ExitCode::FAILURE;
        }
    };

    println!("== chat template");
    println!("   corpus           {mode}");
    println!("   cases            {}", cases.len());
    println!(
        "   tokenizer        {}  vocab {}  added {}  pattern {}",
        tk_info.model_type, tk_info.vocab_size, tk_info.added_tokens, tk_info.pattern_variant
    );
    println!("   markers          {} {} {} {}", chat::IM_START, chat::IM_END, chat::THINK_OPEN, chat::THINK_CLOSE);
    println!();

    // A marker constant is only right if the tokenizer turns it into one id, and
    // that id is the one the checkpoint shipped. Checked once, up front, because
    // every case depends on it.
    let mut marker_failures = 0usize;
    for (marker, want) in [
        (chat::IM_START, 248045u32),
        (chat::IM_END, 248046),
        (chat::THINK_OPEN, 248068),
        (chat::THINK_CLOSE, 248069),
    ] {
        let got = tk.encode(marker);
        if got != vec![want] {
            marker_failures += 1;
            println!("  !! {marker:?} encodes to {got:?}, expected a single id {want}");
        }
    }
    if marker_failures > 0 {
        println!("   RESULT: FAIL (marker constants are not single tokens)");
        return ExitCode::FAILURE;
    }
    println!("   markers are single tokens: 248045 248046 248068 248069");
    println!();

    let mut text_ok = 0usize;
    let mut text_bad = 0usize;
    let mut ids_ok = 0usize;
    let mut ids_bad = 0usize;
    let mut err_ok = 0usize;
    let mut err_bad = 0usize;
    let mut parse_bad = 0usize;
    let mut first_text: Option<String> = None;
    let mut first_ids: Option<String> = None;
    let mut first_err: Option<String> = None;

    for (i, case) in cases.iter().enumerate() {
        if max.is_some_and(|m| i >= m) {
            break;
        }
        let name = case.get("name").and_then(|n| n.as_str()).unwrap_or("?");
        let req = match Request::from_json(case) {
            Ok(r) => r,
            Err(e) => {
                text_bad += 1;
                let msg = format!("case {i} {name:?}: could not be built: {e}");
                println!("  !! {msg}");
                first_text.get_or_insert(msg);
                continue;
            }
        };
        let got = chat::render(&req);

        match (case.get("expect_text").and_then(|t| t.as_str()), &got) {
            (Some(want), Ok(rendered)) if rendered.text == want => text_ok += 1,
            (Some(want), Ok(rendered)) => {
                text_bad += 1;
                let msg = format!(
                    "case {i} {name:?}: text differs\n     want {} bytes\n     got  {} bytes",
                    want.len(),
                    rendered.text.len()
                );
                println!("  !! {msg}");
                if verbose {
                    println!("     want {:?}", truncate(want));
                    println!("     got  {:?}", truncate(&rendered.text));
                }
                first_text.get_or_insert(msg);
            }
            (Some(_), Err(e)) => {
                text_bad += 1;
                let msg =
                    format!("case {i} {name:?}: reference rendered text, this raised {e:?}");
                println!("  !! {msg}");
                first_text.get_or_insert(msg);
            }
            (None, Err(e)) => {
                let want = case.get("expect_error").and_then(|t| t.as_str()).unwrap_or("");
                if e == want {
                    err_ok += 1;
                } else {
                    err_bad += 1;
                    let msg = format!("case {i} {name:?}: error {e:?}, expected {want:?}");
                    println!("  !! {msg}");
                    first_err.get_or_insert(msg);
                }
            }
            (None, Ok(rendered)) => {
                err_bad += 1;
                let want = case.get("expect_error").and_then(|t| t.as_str()).unwrap_or("");
                let msg = format!(
                    "case {i} {name:?}: expected the error {want:?}, got {} bytes of text",
                    rendered.text.len()
                );
                println!("  !! {msg}");
                first_err.get_or_insert(msg);
            }
        }

        // The ids of the rendered text, when the corpus recorded them.
        if let (Some(want), Ok(rendered)) = (case.get("expect_ids").and_then(|v| v.as_array()), &got)
        {
            let want: Vec<u32> = want.iter().filter_map(|v| v.as_usize()).map(|v| v as u32).collect();
            let mine = tk.encode(&rendered.text);
            if mine == want {
                ids_ok += 1;
            } else {
                ids_bad += 1;
                let at = mine.iter().zip(&want).position(|(a, b)| a != b);
                let msg = format!(
                    "case {i} {name:?}: {} ids, expected {}; first difference at {at:?}",
                    mine.len(),
                    want.len()
                );
                println!("  !! {msg}");
                if verbose {
                    println!("     mine {mine:?}");
                    println!("     want {want:?}");
                }
                first_ids.get_or_insert(msg);
            }
        }

        // render -> parse -> render, for the cases that end on an assistant turn,
        // which is the only place a continuation exists to parse.
        let last_is_assistant = req
            .messages
            .last()
            .is_some_and(|m| m.role == chat::Role::Assistant);
        if let (Ok(rendered), true) = (&got, last_is_assistant) {
            if let Some(idx) = rendered.text.rfind("<|im_start|>assistant\n") {
                let tail = &rendered.text[idx + "<|im_start|>assistant\n".len()..];
                if let Some(continuation) = tail.strip_suffix("<|im_end|>\n") {
                    let reply = chatparse::parse_assistant(continuation);
                    let mut back = chat::Message {
                        role: chat::Role::Assistant,
                        content: chat::Content::Text(reply.content.clone()),
                        reasoning: reply.reasoning.clone(),
                        tool_calls: reply.as_chat_calls(),
                    };
                    // The template renders an absent `reasoning_content` and an
                    // empty one identically, so either round-trips.
                    if back.reasoning.as_deref() == Some("") {
                        back.reasoning = Some(String::new());
                    }
                    let mut msgs = req.messages.clone();
                    msgs.pop();
                    msgs.push(back);
                    let again = chat::render(&chat::Request {
                        messages: msgs,
                        tools: req.tools.clone(),
                        opts: req.opts,
                    });
                    let round_tripped = again
                        .ok()
                        .and_then(|r| {
                            r.text.rfind("<|im_start|>assistant\n").map(|j| {
                                r.text[j + "<|im_start|>assistant\n".len()..]
                                    .strip_suffix("<|im_end|>\n")
                                    .unwrap_or_default()
                                    .to_string()
                            })
                        })
                        .map(|t| t == continuation)
                        .unwrap_or(false);
                    if !round_tripped {
                        parse_bad += 1;
                        println!(
                            "  !! case {i} {name:?}: render -> parse -> render changed the bytes"
                        );
                    }
                }
            }
        }
    }

    println!();
    if let Some(m) = &first_text {
        println!("   first text mismatch: {m}");
    }
    if let Some(m) = &first_ids {
        println!("   first id mismatch:   {m}");
    }
    if let Some(m) = &first_err {
        println!("   first error mismatch: {m}");
    }
    println!(
        "   text {text_ok} ok  {text_bad} bad   ids {ids_ok} ok  {ids_bad} bad   \
         errors {err_ok} ok  {err_bad} bad   reparses {parse_bad} bad"
    );
    let ok = text_bad == 0 && ids_bad == 0 && err_bad == 0 && parse_bad == 0;
    println!("   RESULT: {}", if ok { "PASS" } else { "FAIL" });
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn truncate(s: &str) -> String {
    if s.len() <= 200 {
        s.to_string()
    } else {
        format!("{}...", &s[..200])
    }
}
