//! The chat template, implemented branch for branch.
//!
//! `chat_template.jinja` in the checkpoint is 153 lines of Jinja. This is that
//! file, transcribed: no Jinja interpreter, and no subset. Everything the real
//! template emits is emitted here byte for byte, and `chat_corpus.json` (built by
//! `tools/make_chat_corpus.py` from the reference) is the authority on whether
//! that is true.
//!
//! # Why a chat template exists at all
//!
//! The weights contain no notion of "who is speaking". The model sees a flat
//! string, and the roles exist only to decide which markers that string is
//! wrapped in:
//!
//! ```text
//! <|im_start|>system\n{system}<|im_end|>\n
//! <|im_start|>user\n{user}<|im_end|>\n
//! <|im_start|>assistant\n{assistant}<|im_end|>\n
//! ```
//!
//! Two things follow, and both were measured against the checkpoint rather than
//! assumed:
//!
//! * Fed the same question without the markers (`User: What is 2+2?\nAssistant:`),
//!   the model answers `4` and then keeps writing `\nUser:` -- it is *imitating a
//!   transcript*, not taking a turn. With the markers the same question gives
//!   `The answer to "What is 2+2` at p=0.2175, and the confidence itself is the
//!   tell: the correct format is the difference between p=0.9940 and p=0.2175.
//! * This is a reasoning model, so skipping the empty `<think>` block costs
//!   accuracy rather than merely looking odd: without it the first token is
//!   `<think>` (the model starts thinking when it was asked to answer directly)
//!   and `<|im_end|>` is the third-ranked candidate at the very first step, i.e.
//!   it also wants to end the turn.
//!
//! # The two traps
//!
//! * **`|trim` in Jinja is Python's `str.strip`, not Rust's `trim`.** Python's
//!   whitespace set is `White_Space` plus `U+001C..U+001F`, so
//!   `"\x1cHi\x1c"` strips to `Hi` in Python and to itself under
//!   `str::trim`. See [`jinja_trim`].
//! * **`assistant` messages have two shapes.** One *after* the last user query
//!   gets a `<think>` block prepended; one *before* it does not. The boundary is
//!   `last_query_index`, found by scanning backwards and *skipping* any user
//!   message that is itself a wrapped `<tool_response>` -- which is what makes a
//!   multi-step tool loop render correctly.
//!
//! # What is not implemented
//!
//! Nothing in the template. What is absent is *around* it:
//!
//! * Parsing the model's `<tool_call>` output back into arguments — that is
//!   [`crate::chatparse`], because the template only renders.
//! * The vision markers render, but this engine skipped the 153 vision-tower
//!   tensors, so an image part produces ids the language model will read as
//!   ordinary tokens. [`Part::Image`] and [`Part::Video`] therefore exist and
//!   render correctly, and [`render`] says so in its return value rather than
//!   failing silently; see [`Rendered`].

use crate::pyjson::{self, py_str, Value};

/// `<|im_start|>`, id 248045.
pub const IM_START: &str = "<|im_start|>";
/// `<|im_end|>`, id 248046. Also the checkpoint's `eos_token`.
pub const IM_END: &str = "<|im_end|>";
/// `<think>`, id 248068.
pub const THINK_OPEN: &str = "<think>";
/// `</think>`, id 248069.
pub const THINK_CLOSE: &str = "</think>";

const TOOLS_PREFIX: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOLS_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// The exact text the template's `raise_exception` calls carry. Kept as
/// constants because the corpus asserts on them.
pub const ERR_NO_MESSAGES: &str = "No messages provided.";
pub const ERR_SYSTEM_NOT_FIRST: &str = "System message must be at the beginning.";
pub const ERR_NO_USER_QUERY: &str = "No user query found in messages.";
pub const ERR_UNEXPECTED_ROLE: &str = "Unexpected message role.";
pub const ERR_SYSTEM_IMAGE: &str = "System message cannot contain images.";
pub const ERR_SYSTEM_VIDEO: &str = "System message cannot contain videos.";
pub const ERR_ITEM_TYPE: &str = "Unexpected item type in content.";
pub const ERR_CONTENT_TYPE: &str = "Unexpected content type.";

/// A message's role.
///
/// [`Role::Other`] exists because the template distinguishes two failures by
/// *position*: an unknown role before the last user query falls through the
/// backwards scan and produces `No user query found in messages.`, while the same
/// role after it produces `Unexpected message role.`. A closed enum could not
/// express the input that produces the first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
    Other(String),
}

impl Role {
    fn from_name(s: &str) -> Role {
        match s {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => Role::Other(other.to_string()),
        }
    }
}

/// One element of a structured message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Part {
    Text(String),
    Image,
    Video,
    /// Something the template refuses: `Unexpected item type in content.`
    Other,
}

/// A message body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Content {
    /// `content is string`.
    Text(String),
    /// `content is iterable and content is not mapping`.
    Parts(Vec<Part>),
    /// `content is none or content is undefined`.
    None,
    /// A mapping, or a number, or a bool: the template's final `else`.
    Unsupported,
}

/// One `<function=...>` call, with arguments in insertion order.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    /// Ordered, like the Python dict the template iterates with `|items`.
    pub arguments: Vec<(String, Value)>,
}

/// A message.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Content,
    /// `message.reasoning_content`, when it is a string.
    ///
    /// `Some` suppresses the `</think>`-in-content extraction entirely, which is
    /// what the template does -- so `Some("")` and `None` are different inputs.
    pub reasoning: Option<String>,
    pub tool_calls: Vec<FunctionCall>,
}

impl Message {
    pub fn new(role: Role, text: impl Into<String>) -> Message {
        Message {
            role,
            content: Content::Text(text.into()),
            reasoning: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn user(text: impl Into<String>) -> Message {
        Message::new(Role::User, text)
    }

    pub fn assistant(text: impl Into<String>) -> Message {
        Message::new(Role::Assistant, text)
    }

    pub fn system(text: impl Into<String>) -> Message {
        Message::new(Role::System, text)
    }

    pub fn tool(text: impl Into<String>) -> Message {
        Message::new(Role::Tool, text)
    }
}

/// Template options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Options {
    /// Append an empty `assistant` block for the model to fill.
    pub add_generation_prompt: bool,
    /// `true` leaves the block open after `<think>\n` so the model reasons;
    /// `false` closes an *empty* think block, which is "answer directly".
    pub enable_thinking: bool,
    /// Prefix each image/video marker with `Picture N: ` / `Video N: `.
    pub add_vision_id: bool,
}

/// A render request: messages, and the tools the model may call.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Request {
    pub messages: Vec<Message>,
    /// Raw JSON, passed through `tojson` verbatim. The template never inspects
    /// inside a tool definition.
    pub tools: Vec<Value>,
    pub opts: Options,
}

/// The result of a render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    pub text: String,
    /// How many image parts were rendered.
    ///
    /// Non-zero is a warning, not an error: the markers are correct, but this
    /// engine loads the language model without the vision tower, so those ids have
    /// no image behind them. The caller is told rather than left to find out.
    pub images: usize,
    /// How many video parts were rendered.
    pub videos: usize,
}

/// Python's `str.strip`, which is what Jinja's `|trim` filter calls.
///
/// `char::is_whitespace` is Unicode `White_Space`; Python's `str.isspace` is that
/// **plus** `U+001C..U+001F` (the old ASCII "file/group/record/unit separator"
/// controls). Measured, not assumed: the two sets differ by exactly those four
/// code points, and no others. Using `str::trim` here would leave
/// `"\x1cHi\x1c"` alone where the reference strips it to `Hi`.
pub fn jinja_trim(s: &str) -> String {
    s.trim_matches(is_python_space).to_string()
}

fn is_python_space(c: char) -> bool {
    matches!(c, '\u{1c}'..='\u{1f}') || c.is_whitespace()
}

/// Render a conversation the way `chat_template.jinja` does.
///
/// The order of checks is the template's own, and it matters: the backwards
/// `last_query_index` scan runs *before* the main loop, so an unknown role can
/// fail with the scan's error rather than the loop's.
pub fn render(req: &Request) -> Result<Rendered, String> {
    if req.messages.is_empty() {
        return Err(ERR_NO_MESSAGES.to_string());
    }
    let mut out = String::new();
    let mut images = 0usize;
    let mut videos = 0usize;
    let avi = req.opts.add_vision_id;

    // --- the system block: once, before the loop -------------------------
    //
    // With tools, the whole tool description *becomes* the system message, and a
    // real system message is appended to it after a blank line.
    if !req.tools.is_empty() {
        out.push_str(IM_START);
        out.push_str("system\n");
        out.push_str(TOOLS_PREFIX);
        for t in &req.tools {
            out.push('\n');
            out.push_str(&pyjson::dumps(t));
        }
        out.push_str("\n</tools>");
        out.push_str(TOOLS_INSTRUCTIONS);
        if req.messages[0].role == Role::System {
            let c = jinja_trim(&render_content(
                &req.messages[0].content,
                false,
                true,
                avi,
                &mut images,
                &mut videos,
            )?);
            if !c.is_empty() {
                out.push_str("\n\n");
                out.push_str(&c);
            }
        }
        out.push_str(IM_END);
        out.push('\n');
    } else if req.messages[0].role == Role::System {
        let c = jinja_trim(&render_content(
            &req.messages[0].content,
            false,
            true,
            avi,
            &mut images,
            &mut videos,
        )?);
        out.push_str(IM_START);
        out.push_str("system\n");
        out.push_str(&c);
        out.push_str(IM_END);
        out.push('\n');
    }

    // --- last_query_index: backwards, skipping wrapped tool responses -----
    let n = req.messages.len();
    let mut multi_step_tool = true;
    let mut last_query_index = n - 1;
    for i in (0..n).rev() {
        if multi_step_tool && req.messages[i].role == Role::User {
            let c = jinja_trim(&render_content(
                &req.messages[i].content,
                false,
                false,
                avi,
                &mut images,
                &mut videos,
            )?);
            if !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>")) {
                multi_step_tool = false;
                last_query_index = i;
            }
        }
    }
    if multi_step_tool {
        return Err(ERR_NO_USER_QUERY.to_string());
    }

    // --- the main loop ----------------------------------------------------
    for (i, m) in req.messages.iter().enumerate() {
        let mut content = jinja_trim(&render_content(
            &m.content,
            true,
            false,
            avi,
            &mut images,
            &mut videos,
        )?);
        match &m.role {
            // Rendered before the loop; here it is only a position check.
            Role::System => {
                if i != 0 {
                    return Err(ERR_SYSTEM_NOT_FIRST.to_string());
                }
            }
            Role::User => {
                out.push_str(IM_START);
                out.push_str("user\n");
                out.push_str(&content);
                out.push_str(IM_END);
                out.push('\n');
            }
            Role::Assistant => {
                let mut reasoning = String::new();
                if let Some(r) = &m.reasoning {
                    reasoning = r.clone();
                } else if content.contains(THINK_CLOSE) {
                    // `content.split('</think>')[0].rstrip('\n')
                    //      .split('<think>')[-1].lstrip('\n')`
                    let head = content.split(THINK_CLOSE).next().unwrap_or("");
                    let head = head.trim_end_matches('\n');
                    let head = head.rsplit(THINK_OPEN).next().unwrap_or("");
                    reasoning = head.trim_start_matches('\n').to_string();
                    // `content.split('</think>')[-1].lstrip('\n')`
                    let tail = content.rsplit(THINK_CLOSE).next().unwrap_or("");
                    content = tail.trim_start_matches('\n').to_string();
                }
                let reasoning = jinja_trim(&reasoning);

                if i > last_query_index {
                    // Inside the current turn: the think block is present, and may
                    // be empty, which is the "answered directly" shape.
                    out.push_str(IM_START);
                    out.push_str("assistant\n");
                    out.push_str(THINK_OPEN);
                    out.push('\n');
                    out.push_str(&reasoning);
                    out.push('\n');
                    out.push_str(THINK_CLOSE);
                    out.push_str("\n\n");
                    out.push_str(&content);
                } else {
                    out.push_str(IM_START);
                    out.push_str("assistant\n");
                    out.push_str(&content);
                }

                if !m.tool_calls.is_empty() {
                    for (j, tc) in m.tool_calls.iter().enumerate() {
                        // The first call is joined to the content with a blank
                        // line, but only when there *is* content.
                        if j == 0 {
                            if !jinja_trim(&content).is_empty() {
                                out.push_str("\n\n<tool_call>\n<function=");
                            } else {
                                out.push_str("<tool_call>\n<function=");
                            }
                        } else {
                            out.push_str("\n<tool_call>\n<function=");
                        }
                        out.push_str(&tc.name);
                        out.push_str(">\n");
                        for (k, v) in &tc.arguments {
                            out.push_str("<parameter=");
                            out.push_str(k);
                            out.push_str(">\n");
                            out.push_str(&argument_text(v));
                            out.push_str("\n</parameter>\n");
                        }
                        out.push_str("</function>\n</tool_call>");
                    }
                }
                out.push_str(IM_END);
                out.push('\n');
            }
            // Consecutive tool results share *one* user block.
            Role::Tool => {
                if i > 0 && req.messages[i - 1].role != Role::Tool {
                    out.push_str(IM_START);
                    out.push_str("user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(&content);
                out.push_str("\n</tool_response>");
                if i + 1 == n || req.messages[i + 1].role != Role::Tool {
                    out.push_str(IM_END);
                    out.push('\n');
                }
            }
            Role::Other(_) => return Err(ERR_UNEXPECTED_ROLE.to_string()),
        }
    }

    // --- the generation prompt -------------------------------------------
    if req.opts.add_generation_prompt {
        out.push_str(IM_START);
        out.push_str("assistant\n");
        if req.opts.enable_thinking {
            out.push_str(THINK_OPEN);
            out.push('\n');
        } else {
            out.push_str(THINK_OPEN);
            out.push_str("\n\n");
            out.push_str(THINK_CLOSE);
            out.push_str("\n\n");
        }
    }

    Ok(Rendered {
        text: out,
        images,
        videos,
    })
}

/// How the template stringifies one `<parameter>` value.
///
/// `tojson if mapping or (sequence and not string) else str`, i.e. a dict or a
/// list becomes JSON while everything else becomes a bare Python `str` -- so
/// `True` stays `True`, `None` stays `None`, and a string argument is written
/// without quotes.
fn argument_text(v: &Value) -> String {
    match v {
        Value::Array(_) | Value::Object(_) => pyjson::dumps(v),
        other => pyjson::py_str(other),
    }
}

/// `render_content`, the template's macro.
///
/// `do_count` is the macro's `do_vision_count`: the backwards scan calls it with
/// `false` (its result is only tested for `<tool_response>` wrappers, and the
/// image counters must not move), the main loop with `true`.
fn render_content(
    c: &Content,
    do_count: bool,
    is_system: bool,
    add_vision_id: bool,
    images: &mut usize,
    videos: &mut usize,
) -> Result<String, String> {
    match c {
        Content::Text(s) => Ok(s.clone()),
        Content::None => Ok(String::new()),
        Content::Unsupported => Err(ERR_CONTENT_TYPE.to_string()),
        Content::Parts(parts) => {
            let mut out = String::new();
            for p in parts {
                match p {
                    Part::Image => {
                        if is_system {
                            return Err(ERR_SYSTEM_IMAGE.to_string());
                        }
                        if do_count {
                            *images += 1;
                        }
                        if add_vision_id {
                            out.push_str(&format!("Picture {}: ", *images));
                        }
                        out.push_str("<|vision_start|><|image_pad|><|vision_end|>");
                    }
                    Part::Video => {
                        if is_system {
                            return Err(ERR_SYSTEM_VIDEO.to_string());
                        }
                        if do_count {
                            *videos += 1;
                        }
                        if add_vision_id {
                            out.push_str(&format!("Video {}: ", *videos));
                        }
                        out.push_str("<|vision_start|><|video_pad|><|vision_end|>");
                    }
                    Part::Text(t) => out.push_str(t),
                    Part::Other => return Err(ERR_ITEM_TYPE.to_string()),
                }
            }
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// Building a request from JSON
// ---------------------------------------------------------------------------

impl Message {
    /// Read one message the way the template sees it: `role`, `content`,
    /// `reasoning_content`, `tool_calls`.
    pub fn from_json(v: &Value) -> Result<Message, String> {
        let role = v
            .get("role")
            .and_then(|r| r.as_str())
            .ok_or_else(|| "message has no string `role`".to_string())?;
        let content = match v.get("content") {
            None => Content::None,
            Some(c) => content_from_json(c)?,
        };
        let reasoning = match v.get("reasoning_content") {
            Some(Value::Str(s)) => Some(s.clone()),
            _ => None,
        };
        let mut tool_calls = Vec::new();
        if let Some(Value::Array(calls)) = v.get("tool_calls") {
            for c in calls {
                // `tool_call.function if defined else tool_call`
                let f = c.get("function").unwrap_or(c);
                let name = f
                    .get("name")
                    .and_then(|n| n.as_str())
                    .ok_or_else(|| "a tool call has no string `name`".to_string())?;
                let mut arguments = Vec::new();
                if let Some(Value::Object(kv)) = f.get("arguments") {
                    for (k, val) in kv {
                        arguments.push((k.clone(), val.clone()));
                    }
                }
                tool_calls.push(FunctionCall {
                    name: name.to_string(),
                    arguments,
                });
            }
        }
        Ok(Message {
            role: Role::from_name(role),
            content,
            reasoning,
            tool_calls,
        })
    }
}

fn content_from_json(c: &Value) -> Result<Content, String> {
    match c {
        Value::Str(s) => Ok(Content::Text(s.clone())),
        Value::Null => Ok(Content::None),
        Value::Array(items) => {
            let mut parts = Vec::with_capacity(items.len());
            // The template tests `image`, then `video`, then `text`, then fails.
            for item in items {
                if item.contains_key("image") || item.contains_key("image_url") || type_is(item, "image") {
                    parts.push(Part::Image);
                } else if item.contains_key("video") || type_is(item, "video") {
                    parts.push(Part::Video);
                } else if let Some(t) = item.get("text") {
                    // Jinja renders `{{ item.text }}` with Python's `str`, so a
                    // non-string here is not an error: `{"text": 5}` renders `5`
                    // and `{"text": ["a"]}` renders `['a']`.
                    parts.push(Part::Text(py_str(t)));
                } else {
                    parts.push(Part::Other);
                }
            }
            Ok(Content::Parts(parts))
        }
        // A mapping, a number, a bool: `Unexpected content type.`
        _ => Ok(Content::Unsupported),
    }
}

fn type_is(item: &Value, want: &str) -> bool {
    item.get("type").and_then(|t| t.as_str()) == Some(want)
}

impl Request {
    /// Build a request from `{"messages": [...], "tools": [...], ...}`.
    pub fn from_json(v: &Value) -> Result<Request, String> {
        let messages = match v.get("messages") {
            Some(Value::Array(a)) => a
                .iter()
                .map(Message::from_json)
                .collect::<Result<Vec<_>, _>>()?,
            Some(_) => return Err("`messages` is not an array".to_string()),
            None => Vec::new(),
        };
        let tools = match v.get("tools") {
            Some(Value::Array(a)) => a.clone(),
            _ => Vec::new(),
        };
        let opts = Options {
            add_generation_prompt: v
                .get("add_generation_prompt")
                .and_then(|b| b.as_bool())
                .unwrap_or(false),
            enable_thinking: v
                .get("enable_thinking")
                .and_then(|b| b.as_bool())
                .unwrap_or(false),
            add_vision_id: v.get("add_vision_id").and_then(|b| b.as_bool()).unwrap_or(false),
        };
        Ok(Request {
            messages,
            tools,
            opts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(req: &Request) -> String {
        render(req).unwrap().text
    }

    fn opts(gen: bool) -> Options {
        Options {
            add_generation_prompt: gen,
            ..Options::default()
        }
    }

    #[test]
    fn single_turn() {
        let r = Request {
            messages: vec![Message::user("Hi")],
            tools: vec![],
            opts: opts(true),
        };
        assert_eq!(
            text(&r),
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn system_message_comes_first_and_in_its_own_block() {
        let r = Request {
            messages: vec![Message::system("You are terse."), Message::user("Hi")],
            tools: vec![],
            opts: opts(true),
        };
        assert_eq!(
            text(&r),
            "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    /// Only an assistant message *after* the last user query gets a think block.
    #[test]
    fn think_block_is_position_dependent() {
        let r = Request {
            messages: vec![
                Message::user("a"),
                Message::assistant("b"),
                Message::user("c"),
                Message::assistant("d"),
            ],
            tools: vec![],
            opts: opts(false),
        };
        assert_eq!(
            text(&r),
            "<|im_start|>user\na<|im_end|>\n\
             <|im_start|>assistant\nb<|im_end|>\n\
             <|im_start|>user\nc<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\nd<|im_end|>\n"
        );
    }

    /// A user message that is itself a wrapped tool response does not end the
    /// search, so an assistant message *before* it is still "inside the current
    /// turn" and gets a think block. Without the skip, `last_query_index` would be
    /// that message's index and the assistant would lose its block.
    #[test]
    fn multi_step_tool_query_is_skipped() {
        let r = Request {
            messages: vec![
                Message::user("q"),
                Message::assistant("a"),
                Message::user("<tool_response>\nR\n</tool_response>"),
            ],
            tools: vec![],
            opts: opts(false),
        };
        assert_eq!(
            text(&r),
            "<|im_start|>user\nq<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\na<|im_end|>\n\
             <|im_start|>user\n<tool_response>\nR\n</tool_response><|im_end|>\n"
        );

        // With no unwrapped query anywhere the scan runs out, and that is an
        // error -- the reference does the same, not just this implementation.
        let bad = Request {
            messages: vec![Message::user("<tool_response>\nR\n</tool_response>")],
            tools: vec![],
            opts: opts(true),
        };
        assert_eq!(render(&bad).unwrap_err(), ERR_NO_USER_QUERY);
    }

    #[test]
    fn consecutive_tool_results_share_one_user_block() {
        let r = Request {
            messages: vec![
                Message::user("q"),
                Message::tool("R1"),
                Message::tool("R2"),
            ],
            tools: vec![],
            opts: opts(true),
        };
        assert_eq!(
            text(&r),
            "<|im_start|>user\nq<|im_end|>\n\
             <|im_start|>user\n<tool_response>\nR1\n</tool_response>\n<tool_response>\nR2\n</tool_response><|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn tool_calls_render_as_xml_and_keep_argument_types() {
        let r = Request {
            messages: vec![
                Message::user("q"),
                Message {
                    role: Role::Assistant,
                    content: Content::Text(String::new()),
                    reasoning: None,
                    tool_calls: vec![FunctionCall {
                        name: "f".into(),
                        arguments: vec![
                            ("s".into(), Value::Str("str".into())),
                            ("i".into(), Value::Int(42)),
                            ("f".into(), Value::Float(1.5)),
                            ("b".into(), Value::Bool(true)),
                            ("n".into(), Value::Null),
                            ("lst".into(), Value::Array(vec![Value::Int(1), Value::Int(2)])),
                            // A nested dict is `tojson` in Python's own format.
                            (
                                "d".into(),
                                Value::Object(vec![("x".into(), Value::Int(1))]),
                            ),
                        ],
                    }],
                },
            ],
            tools: vec![],
            opts: opts(false),
        };
        // Empty content: no blank line before the call.
        assert_eq!(
            text(&r),
            "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n\
             <tool_call>\n<function=f>\n\
             <parameter=s>\nstr\n</parameter>\n\
             <parameter=i>\n42\n</parameter>\n\
             <parameter=f>\n1.5\n</parameter>\n\
             <parameter=b>\nTrue\n</parameter>\n\
             <parameter=n>\nNone\n</parameter>\n\
             <parameter=lst>\n[1, 2]\n</parameter>\n\
             <parameter=d>\n{\"x\": 1}\n</parameter>\n\
             </function>\n</tool_call><|im_end|>\n"
        );
    }

    /// With content, the call is separated by a blank line; with two calls the
    /// second is separated by one newline.
    #[test]
    fn tool_call_separation_depends_on_content() {
        let call = |n: &str| FunctionCall {
            name: n.into(),
            arguments: vec![],
        };
        let mut m = Message::assistant("thinking about it");
        m.tool_calls = vec![call("a"), call("b")];
        let r = Request {
            messages: vec![Message::user("q"), m],
            tools: vec![],
            opts: opts(false),
        };
        assert_eq!(
            text(&r),
            "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n\
             thinking about it\n\n<tool_call>\n<function=a>\n</function>\n</tool_call>\
             \n<tool_call>\n<function=b>\n</function>\n</tool_call><|im_end|>\n"
        );
    }

    #[test]
    fn reasoning_content_wins_over_inline_think() {
        let mut m = Message::assistant("inline\n</think>\n\ntail");
        m.reasoning = Some("explicit".into());
        let r = Request {
            messages: vec![Message::user("q"), m],
            tools: vec![],
            opts: opts(false),
        };
        assert_eq!(
            text(&r),
            "<|im_start|>user\nq<|im_end|>\n\
             <|im_start|>assistant\n<think>\nexplicit\n</think>\n\ninline\n</think>\n\ntail<|im_end|>\n"
        );
    }

    #[test]
    fn inline_think_is_split_out() {
        let mut m = Message::assistant("<think>\nrz\n</think>\n\nans");
        m.reasoning = None;
        let r = Request {
            messages: vec![Message::user("q"), m],
            tools: vec![],
            opts: opts(false),
        };
        assert_eq!(
            text(&r),
            "<|im_start|>user\nq<|im_end|>\n\
             <|im_start|>assistant\n<think>\nrz\n</think>\n\nans<|im_end|>\n"
        );
    }

    #[test]
    fn thinking_toggle_changes_the_generation_prompt() {
        let base = Request {
            messages: vec![Message::user("Hi")],
            tools: vec![],
            opts: opts(true),
        };
        let mut on = base.clone();
        on.opts.enable_thinking = true;
        assert!(text(&on).ends_with("<|im_start|>assistant\n<think>\n"));
        assert!(!text(&on).contains("</think>"));
        assert!(text(&base).ends_with("<think>\n\n</think>\n\n"));
    }

    /// Python's whitespace set is `White_Space` plus `U+001C..U+001F`, so a Rust
    /// `trim` here would leave four code points the reference removes.
    #[test]
    fn trim_is_pythons_strip_not_rusts() {
        assert_eq!(jinja_trim("  Hi  \n"), "Hi");
        assert_eq!(jinja_trim("\u{1c}Hi\u{1c}"), "Hi");
        assert_eq!(jinja_trim("\u{1f}Hi"), "Hi");
        assert_eq!(jinja_trim("\u{85}Hi\u{a0}"), "Hi");
        assert_eq!(jinja_trim("\u{3000}Hi\u{3000}"), "Hi");
        // Rust's own trim disagrees, which is the whole point.
        assert_ne!("\u{1c}Hi\u{1c}".trim(), "Hi");
        // And an ordinary character is not touched.
        assert_eq!(jinja_trim("\u{200b}Hi"), "\u{200b}Hi");
    }

    #[test]
    fn vision_parts_render_and_count() {
        let mk = |parts: Vec<Part>, avi: bool| Request {
            messages: vec![Message {
                role: Role::User,
                content: Content::Parts(parts),
                reasoning: None,
                tool_calls: Vec::new(),
            }],
            tools: vec![],
            opts: Options {
                add_generation_prompt: true,
                enable_thinking: false,
                add_vision_id: avi,
            },
        };
        let r = mk(
            vec![Part::Image, Part::Image, Part::Text("t".into())],
            false,
        );
        let got = render(&r).unwrap();
        assert_eq!(got.images, 2);
        assert!(got.text.starts_with(
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\
             <|vision_start|><|image_pad|><|vision_end|>t<|im_end|>\n"
        ));
        let with_ids = mk(vec![Part::Image, Part::Image, Part::Text("t".into())], true);
        assert!(render(&with_ids)
            .unwrap()
            .text
            .contains("Picture 1: <|vision_start|><|image_pad|><|vision_end|>Picture 2: "));
        let video = mk(vec![Part::Video], false);
        assert_eq!(render(&video).unwrap().videos, 1);
    }

    #[test]
    fn content_shapes_that_fail() {
        // The bad content goes on the message whose role is being tested; a
        // system message needs a user after it, a user message does not.
        let with = |role: Role, c: Content| {
            let bad = Message {
                role: role.clone(),
                content: c,
                reasoning: None,
                tool_calls: Vec::new(),
            };
            let messages = if role == Role::System {
                vec![bad, Message::user("q")]
            } else {
                vec![Message::system("s"), bad, Message::user("q")]
            };
            Request {
                messages,
                tools: vec![],
                opts: opts(true),
            }
        };
        // A mapping is not a valid content type.
        assert_eq!(
            render(&with(Role::User, Content::Unsupported)).unwrap_err(),
            ERR_CONTENT_TYPE
        );
        assert_eq!(
            render(&with(Role::System, Content::Unsupported)).unwrap_err(),
            ERR_CONTENT_TYPE
        );
        // An image inside a system message is refused, with its own message.
        assert_eq!(
            render(&with(Role::System, Content::Parts(vec![Part::Image]))).unwrap_err(),
            ERR_SYSTEM_IMAGE
        );
        assert_eq!(
            render(&with(Role::System, Content::Parts(vec![Part::Video]))).unwrap_err(),
            ERR_SYSTEM_VIDEO
        );
        // An item that is none of image/video/text.
        assert_eq!(
            render(&with(Role::User, Content::Parts(vec![Part::Other]))).unwrap_err(),
            ERR_ITEM_TYPE
        );
        // And the same list on a *user* message is fine, so the refusal above is
        // about the role rather than about the part.
        assert!(render(&with(Role::User, Content::Parts(vec![Part::Image]))).is_ok());
    }

    #[test]
    fn position_and_role_errors() {
        let sys_late = Request {
            messages: vec![Message::user("q"), Message::system("s")],
            tools: vec![],
            opts: opts(true),
        };
        assert_eq!(render(&sys_late).unwrap_err(), ERR_SYSTEM_NOT_FIRST);

        // Unknown role *after* the query: the loop's error.
        let bad = Request {
            messages: vec![Message::user("q"), Message::new(Role::Other("wizard".into()), "x")],
            tools: vec![],
            opts: opts(true),
        };
        assert_eq!(render(&bad).unwrap_err(), ERR_UNEXPECTED_ROLE);

        // Unknown role as the *only* message: the backwards scan never finds a
        // user, so this is the scan's error, not the loop's.
        let only = Request {
            messages: vec![Message::new(Role::Other("wizard".into()), "x")],
            tools: vec![],
            opts: opts(true),
        };
        assert_eq!(render(&only).unwrap_err(), ERR_NO_USER_QUERY);

        assert_eq!(render(&Request::default()).unwrap_err(), ERR_NO_MESSAGES);
    }

    #[test]
    fn a_tool_block_still_needs_a_user_query() {
        // The tool block forces a system message out, but there is no user
        // message, so the backwards scan fails.
        let r = Request {
            messages: vec![Message::system("s")],
            tools: vec![Value::Object(vec![])],
            opts: opts(true),
        };
        assert_eq!(render(&r).unwrap_err(), ERR_NO_USER_QUERY);
    }

    /// The tool block is the whole system message, and a real system message is
    /// appended to it after a blank line -- both shapes, since an empty system
    /// message must not leave a stray blank line.
    #[test]
    fn tools_block_shapes() {
        let tool = Value::Object(vec![(
            "function".into(),
            Value::Object(vec![("name".into(), Value::Str("f".into()))]),
        )]);
        let with = |msgs: Vec<Message>, tools: Vec<Value>| Request {
            messages: msgs,
            tools,
            opts: opts(true),
        };
        let a = text(&with(vec![Message::user("q")], vec![tool.clone()]));
        assert!(a.starts_with("<|im_start|>system\n# Tools\n\nYou have access"));
        assert!(a.contains("\n{\"function\": {\"name\": \"f\"}}\n</tools>"));
        assert!(a.contains("</IMPORTANT><|im_end|>\n<|im_start|>user\nq"));
        assert!(!a.contains("</IMPORTANT>\n\n"), "no system message: no blank line");

        let b = text(&with(
            vec![Message::system("You are helpful."), Message::user("q")],
            vec![tool.clone()],
        ));
        assert!(b.contains("</IMPORTANT>\n\nYou are helpful.<|im_end|>\n"));

        // An empty system message contributes nothing, not two newlines.
        let c = text(&with(
            vec![Message::system("   "), Message::user("q")],
            vec![tool.clone()],
        ));
        assert!(c.contains("</IMPORTANT><|im_end|>\n"));

        // Two tools: one JSON document per line.
        let d = text(&with(vec![Message::user("q")], vec![tool.clone(), tool]));
        assert!(d.contains("\n{\"function\": {\"name\": \"f\"}}\n{\"function\": {\"name\": \"f\"}}\n</tools>"));
    }
}
