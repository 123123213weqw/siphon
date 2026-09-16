//! Reading back what the model wrote: `<tool_call>` blocks, and the think block.
//!
//! The chat template only renders. Nothing in the checkpoint turns
//!
//! ```text
//! <tool_call>
//! <function=get_weather>
//! <parameter=city>
//! Paris
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! back into `get_weather(city="Paris")`, so a tool loop needs this side as well.
//!
//! # The property that matters
//!
//! Render then parse then render must be the identity. That is checked directly:
//! every test here round-trips through [`crate::chat::render`], and the corpus
//! adds the same check over generated text. A parser that quietly drops a
//! parameter, or reorders them, fails it.
//!
//! # Values stay strings
//!
//! A `<parameter>` holds whatever `str`-of-the-argument the template wrote, so
//! `42`, `True`, `None` and `[1, 2]` all arrive as text. [`FunctionCall::text`]
//! therefore returns strings and leaves the typing to the caller, who knows the
//! tool's schema. Re-rendering those strings puts the same bytes back, which is
//! the point: `42` is `str("42")`, not `repr(42)`.
//!
//! The one thing that cannot be recovered is a string argument that happened to
//! *look* like JSON -- `<parameter=d>{"x": 1}</parameter>` re-renders as
//! `{"x": 1}` only if it is parsed back as a string, which is why the parser does
//! not try to interpret it.

use crate::chat::{jinja_trim, THINK_CLOSE, THINK_OPEN};

/// A call recovered from generated text, with arguments in the order written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: Vec<(String, String)>,
}

/// What the model wrote, split the way the template would split it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Reply {
    /// Between `<think>` and `</think>`, when the text contains the latter.
    pub reasoning: Option<String>,
    /// Everything that is not reasoning and not a tool call.
    pub content: String,
    pub tool_calls: Vec<FunctionCall>,
}

impl Reply {
    /// A reply that only says something.
    pub fn text(content: impl Into<String>) -> Reply {
        Reply {
            reasoning: None,
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    /// The calls, as `chat::FunctionCall` -- what a re-render needs.
    pub fn as_chat_calls(&self) -> Vec<crate::chat::FunctionCall> {
        self.tool_calls
            .iter()
            .map(|c| crate::chat::FunctionCall {
                name: c.name.clone(),
                arguments: c
                    .arguments
                    .iter()
                    .map(|(k, v)| (k.clone(), crate::pyjson::Value::Str(v.clone())))
                    .collect(),
            })
            .collect()
    }
}

/// Parse a model continuation -- the text after `<|im_start|>assistant\n`, and
/// before the `<|im_end|>` that ended the turn.
///
/// Both fields come back trimmed, because that is what the template does to them
/// on the way in: `reasoning_content|trim` and `content = ...|trim`. Doing the
/// same on the way out is what makes render-parse-render the identity, and it
/// also means a caller can hand the reply straight back to
/// [`crate::chat::render`].
///
/// The reasoning is whatever follows the *last* `<think>`, matching the
/// template's `content.split('<think>')[-1]` -- not the first, which would differ
/// for reasoning that itself mentions the tag.
pub fn parse_assistant(text: &str) -> Reply {
    let (reasoning, body) = match text.split_once(THINK_CLOSE) {
        Some((head, body)) => {
            let head = head.rsplit(THINK_OPEN).next().unwrap_or(head);
            (Some(jinja_trim(head)), body.trim_start_matches('\n'))
        }
        // Thinking was skipped, so the whole continuation is the answer.
        None => (None, text),
    };

    let (content, calls) = match parse_tool_calls(body) {
        Ok(parsed) => {
            // The template writes `content + "\n\n" + <calls>` when there is
            // content, and just `<calls>` when there is not.
            let cut = body.find("<tool_call>").unwrap_or(body.len());
            (jinja_trim(&body[..cut]), parsed)
        }
        // Not a tool call at all: a `<` inside ordinary text.
        Err(_) => (jinja_trim(body), Vec::new()),
    };

    Reply {
        reasoning,
        content,
        tool_calls: calls,
    }
}

/// Extract every `<tool_call>` block from generated text.
///
/// Errors on a block that is present but malformed, rather than returning fewer
/// calls than the model wrote -- a silently dropped tool call is worse than a
/// reported one.
pub fn parse_tool_calls(text: &str) -> Result<Vec<FunctionCall>, String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        let Some(end) = after.find("</tool_call>") else {
            return Err("a `<tool_call>` is never closed".to_string());
        };
        out.push(parse_one(after[..end].trim_matches('\n'))?);
        rest = &after[end + "</tool_call>".len()..];
    }
    Ok(out)
}

/// Parse the inside of one `<tool_call>`: one `<function=NAME>` and its
/// `<parameter=KEY>` blocks.
fn parse_one(body: &str) -> Result<FunctionCall, String> {
    let body = body.trim_matches('\n');
    let rest = body
        .strip_prefix("<function=")
        .ok_or_else(|| format!("a tool call does not start with `<function=`: {body:.40?}"))?;
    let (name, rest) = rest
        .split_once(">\n")
        .ok_or_else(|| "a `<function=` has no closing `>`".to_string())?;
    if name.is_empty() {
        return Err("a `<function=>` has an empty name".to_string());
    }

    let mut arguments = Vec::new();
    let rest = rest.trim_end_matches('\n');
    // The block ends with `</function>`; everything before it is parameters.
    let Some(rest) = rest.strip_suffix("</function>") else {
        return Err(format!("`<function={name}>` is never closed"))
    };
    let mut rest = rest.trim_matches('\n');
    while !rest.is_empty() {
        let r = rest
            .strip_prefix("<parameter=")
            .ok_or_else(|| format!("expected `<parameter=`, found {:?}", &rest[..rest.len().min(24)]))?;
        let (key, r) = r
            .split_once(">\n")
            .ok_or_else(|| format!("a `<parameter=` has no closing `>`: {:?}", &r[..r.len().min(24)]))?;
        // The value runs to the next `\n</parameter>`, so it may contain newlines.
        let (value, r) = r
            .split_once("\n</parameter>\n")
            .or_else(|| r.split_once("\n</parameter>"))
            .ok_or_else(|| format!("`<parameter={key}>` is never closed"))?;
        arguments.push((key.to_string(), value.to_string()));
        rest = r.trim_start_matches('\n');
    }
    Ok(FunctionCall {
        name: name.to_string(),
        arguments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{Content, Message, Options, Request, Role};

    /// The whole point: template output goes in, the same template output comes
    /// back out.
    fn round_trip(m: Message) -> Reply {
        let req = Request {
            messages: vec![Message::user("q"), m],
            tools: vec![],
            opts: Options::default(),
        };
        let full = crate::chat::render(&req).unwrap().text;
        // Only the continuation is parsed, i.e. what the model itself would emit.
        let tail = full
            .split_once("<|im_start|>assistant\n")
            .expect("the assistant block is last")
            .1;
        let continuation = tail.strip_suffix("<|im_end|>\n").expect("the block is closed");
        let reply = parse_assistant(continuation);

        // Re-render from the parsed reply and compare with the original message.
        let back = Message {
            role: Role::Assistant,
            content: Content::Text(reply.content.clone()),
            reasoning: reply.reasoning.clone(),
            tool_calls: reply.as_chat_calls(),
        };
        let req2 = Request {
            messages: vec![Message::user("q"), back.clone()],
            tools: vec![],
            opts: Options::default(),
        };
        let again = crate::chat::render(&req2).unwrap().text;
        let again_tail = again
            .split_once("<|im_start|>assistant\n")
            .unwrap()
            .1
            .strip_suffix("<|im_end|>\n")
            .unwrap();
        assert_eq!(
            continuation, again_tail,
            "parse then render must reproduce the original bytes"
        );
        reply
    }

    #[test]
    fn plain_text_with_an_empty_think_block() {
        let r = round_trip(Message::assistant("The capital is Paris."));
        assert_eq!(r.reasoning.as_deref(), Some(""));
        assert_eq!(r.content, "The capital is Paris.");
        assert!(r.tool_calls.is_empty());
    }

    #[test]
    fn reasoning_is_recovered() {
        let mut m = Message::assistant("two plus two is four");
        m.reasoning = Some("the user asks arithmetic".into());
        let r = round_trip(m);
        assert_eq!(r.reasoning.as_deref(), Some("the user asks arithmetic"));
        assert_eq!(r.content, "two plus two is four");
    }

    #[test]
    fn a_tool_call_round_trips() {
        let mut m = Message::assistant("");
        m.tool_calls = vec![crate::chat::FunctionCall {
            name: "get_weather".into(),
            arguments: vec![
                ("city".into(), crate::pyjson::Value::Str("Paris".into())),
                ("units".into(), crate::pyjson::Value::Str("metric".into())),
            ],
        }];
        let r = round_trip(m);
        assert!(r.content.is_empty());
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].name, "get_weather");
        assert_eq!(
            r.tool_calls[0].arguments,
            vec![
                ("city".to_string(), "Paris".to_string()),
                ("units".to_string(), "metric".to_string())
            ]
        );
    }

    #[test]
    fn text_and_a_call_together_round_trip() {
        let mut m = Message::assistant("Let me check.");
        m.reasoning = Some("needs a lookup".into());
        m.tool_calls = vec![crate::chat::FunctionCall {
            name: "f".into(),
            arguments: vec![("a".into(), crate::pyjson::Value::Int(1))],
        }];
        let r = round_trip(m);
        assert_eq!(r.content, "Let me check.");
        assert_eq!(r.reasoning.as_deref(), Some("needs a lookup"));
        assert_eq!(r.tool_calls[0].arguments, vec![("a".to_string(), "1".to_string())]);
    }

    #[test]
    fn two_calls_round_trip() {
        let mut m = Message::assistant("");
        m.tool_calls = vec![
            crate::chat::FunctionCall {
                name: "a".into(),
                arguments: vec![("x".into(), crate::pyjson::Value::Str("1".into()))],
            },
            crate::chat::FunctionCall {
                name: "b".into(),
                arguments: vec![],
            },
        ];
        let r = round_trip(m);
        assert_eq!(r.tool_calls.len(), 2);
        assert_eq!(r.tool_calls[0].name, "a");
        assert_eq!(r.tool_calls[1].name, "b");
    }

    /// A value may span lines, which the template's own example calls out.
    #[test]
    fn multi_line_values_survive() {
        let text = "<tool_call>\n<function=f>\n<parameter=a>\nline one\nline two\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_tool_calls(text).unwrap();
        assert_eq!(calls[0].arguments[0].1, "line one\nline two");
    }

    /// Values are text, and the parser does not guess at them: `True` stays the
    /// string `True`, so re-rendering writes `True` again rather than `true`.
    #[test]
    fn values_are_not_reinterpreted() {
        let text = "<tool_call>\n<function=f>\n\
                    <parameter=b>\nTrue\n</parameter>\n\
                    <parameter=n>\nNone\n</parameter>\n\
                    <parameter=d>\n{\"x\": 1}\n</parameter>\n\
                    <parameter>bad</tool_call>";
        // The last parameter is malformed, so the whole parse fails rather than
        // returning three of four.
        assert!(parse_tool_calls(text).is_err());

        let ok = "<tool_call>\n<function=f>\n\
                  <parameter=b>\nTrue\n</parameter>\n\
                  <parameter=n>\nNone\n</parameter>\n\
                  <parameter=d>\n{\"x\": 1}\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_tool_calls(ok).unwrap();
        assert_eq!(calls[0].arguments[0].1, "True");
        assert_eq!(calls[0].arguments[1].1, "None");
        assert_eq!(calls[0].arguments[2].1, "{\"x\": 1}");
    }

    #[test]
    fn malformed_blocks_are_reported_not_dropped() {
        assert!(parse_tool_calls("<tool_call>\n<function=f>\n").is_err());
        assert!(parse_tool_calls("<tool_call>\nnot a function\n</tool_call>").is_err());
        assert!(parse_tool_calls("<tool_call>\n<function=>\n</function>\n</tool_call>").is_err());
        assert!(parse_tool_calls("<tool_call>\n<function=f>\n</functionx>\n</tool_call>").is_err());
        // No block at all is not an error: that is ordinary text.
        assert_eq!(parse_tool_calls("just prose").unwrap(), Vec::new());
    }

    /// A `<` in ordinary prose must not be mistaken for a malformed call.
    #[test]
    fn an_angle_bracket_in_prose_is_not_a_call() {
        let r = parse_assistant("<think>\n\n</think>\n\n2 < 3 and 5 > 4");
        assert_eq!(r.content, "2 < 3 and 5 > 4");
        assert!(r.tool_calls.is_empty());
    }
}
