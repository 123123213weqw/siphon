//! A small JSON reader and writer whose output matches Python's `json.dumps`.
//!
//! Why not `serde_json`: the chat template pipes tool definitions through Jinja's
//! `tojson`, which is `json.dumps(obj, ensure_ascii=False)`. Byte-exactness is the
//! whole point of the comparison against the reference, and `serde_json` differs
//! from Python in three ways that all show up in ordinary tool definitions:
//!
//! * **Separators.** Python writes `{"a": 1, "b": 2}`; `serde_json` writes
//!   `{"a":1,"b":2}`.
//! * **Key order.** `serde_json::Map` is a `BTreeMap` by default, so keys come out
//!   sorted; Python preserves insertion order. A tool's `{"name", "description",
//!   "parameters"}` would be reordered to `description, name, parameters`.
//! * **Float formatting.** Python uses `repr(float)`, which switches to exponent
//!   notation outside `[1e-4, 1e16)` and pads the exponent to two digits: `1e-05`,
//!   `1e+16`. Rust's `{}` gives `0.00001` and `10000000000000000`.
//!
//! So this module keeps object keys in insertion order, and carries `repr` and
//! `str` of every value rather than a single "to string".
//!
//! # `dumps` is not `str`
//!
//! The template uses both, chosen per argument by type:
//!
//! ```text
//! args_value | tojson   if it is a mapping, or a sequence that is not a string
//! args_value | string   otherwise
//! ```
//!
//! which is why `True` comes out as `True` (not `true`) and `None` as `None` (not
//! `null`) in a rendered `<parameter=...>`, while a nested dict comes out as JSON.
//! Both are implemented here; see [`dumps`] and [`py_str`].

use std::fmt::Write as _;

/// A JSON value that remembers the order its object keys were written in.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// Integers are kept separately from floats because Python's `str(1)` is `1`
    /// while `str(1.0)` is `1.0`, and a `<parameter>` renders a bare `str`.
    Int(i128),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    /// Look up an object key. `None` for anything that is not an object, and for a
    /// key that is absent -- the same `Undefined` a Jinja template would see.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Whether an object has a key. Jinja's `'k' in obj`, which for a *string* is a
    /// substring test -- handled here because the template applies it to messages
    /// whose shape is not guaranteed.
    pub fn contains_key(&self, key: &str) -> bool {
        match self {
            Value::Object(kv) => kv.iter().any(|(k, _)| k == key),
            Value::Str(s) => s.contains(key),
            _ => false,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Value::Int(i) => usize::try_from(*i).ok(),
            _ => None,
        }
    }

    /// A non-empty array, which is what `tools and tools is iterable and tools is
    /// not mapping` amounts to. `None` when the key is missing, not an array, or
    /// empty.
    pub fn as_nonempty_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) if !a.is_empty() => Some(a),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Python's `repr(float)`.
///
/// The digits come from Rust's `{:e}`, which is also shortest-round-trip, so the
/// two agree on the digit string; what differs is the *layout*, which Python
/// decides by one rule:
///
/// ```text
/// exponent form  iff  decpt <= -4  or  decpt > 16
/// ```
///
/// where `decpt` is the decimal point's position in `0.<digits> * 10**decpt`. That
/// single rule reproduces all of `1e-05`, `0.0001`, `1e+16`,
/// `1000000000000000.0`, `100.0`, `3.0` and `-0.0`.
pub fn py_repr_f64(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    let s = format!("{x:e}");
    let (mant, exp) = s.split_once('e').expect("`{:e}` always writes an exponent");
    let exp: i32 = exp.parse().expect("`{:e}` writes a decimal exponent");
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mant),
    };
    // Rust's `{:e}` writes `d[.ddd]`; flatten to a pure digit string.
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1;

    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        out.push(digits.chars().next().unwrap_or('0'));
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        let e = decpt - 1;
        out.push(if e < 0 { '-' } else { '+' });
        let mag = e.unsigned_abs();
        if mag < 10 {
            out.push('0');
        }
        let _ = write!(out, "{mag}");
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..-decpt {
            out.push('0');
        }
        out.push_str(&digits);
    } else if decpt as usize >= digits.len() {
        out.push_str(&digits);
        for _ in 0..(decpt as usize - digits.len()) {
            out.push('0');
        }
        // `Py_DTSF_ADD_DOT_0`: a float always shows a fractional part.
        out.push_str(".0");
    } else {
        let cut = decpt as usize;
        out.push_str(&digits[..cut]);
        out.push('.');
        out.push_str(&digits[cut..]);
    }
    out
}

/// Whether a character survives Python's `json.dumps` untouched.
///
/// Python escapes `"`, `\` and the C0 controls, and nothing else. In particular
/// `\x7f`, `U+2028` and `U+2029` are written literally, which is where a
/// JavaScript-style escaper (or `serde_json`'s `\u007f`-free but `<`-escaping
/// sibling in Jinja proper) would diverge.
fn json_escape_into(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

/// Python's `json.dumps(v, ensure_ascii=False)`.
///
/// `ensure_ascii=False` is not an assumption: it was measured against the
/// reference (`json.dumps({"k": "中文"})` keeps the characters, and `<a>&'</a>`
/// comes through unescaped, so there is no HTML-safe escaping either).
pub fn dumps(v: &Value) -> String {
    let mut out = String::new();
    dumps_into(v, &mut out);
    out
}

fn dumps_into(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Value::Float(f) => out.push_str(&py_repr_f64(*f)),
        Value::Str(s) => {
            out.push('"');
            json_escape_into(s, out);
            out.push('"');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, e) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dumps_into(e, out);
            }
            out.push(']');
        }
        Value::Object(kv) => {
            out.push('{');
            for (i, (k, e)) in kv.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push('"');
                json_escape_into(k, out);
                out.push_str("\": ");
                dumps_into(e, out);
            }
            out.push('}');
        }
    }
}

/// Python's `str(v)`, which for a container is its `repr`.
///
/// The chat template only reaches this for scalars -- its filter is
/// `tojson if mapping or (sequence and not string) else string` -- but the
/// container branches are implemented anyway so that no call shape is silently
/// wrong. Note the difference from [`dumps`]: single quotes, and `True`/`None`
/// rather than `true`/`null`.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => py_repr_f64(*f),
        Value::Str(s) => s.clone(),
        Value::Array(a) => {
            let mut out = String::from("[");
            for (i, e) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&py_repr(e));
            }
            out.push(']');
            out
        }
        Value::Object(kv) => {
            let mut out = String::from("{");
            for (i, (k, e)) in kv.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&py_repr_str(k));
                out.push_str(": ");
                out.push_str(&py_repr(e));
            }
            out.push('}');
            out
        }
    }
}

/// Python's `repr` of a value, i.e. `str` except that a string is quoted.
fn py_repr(v: &Value) -> String {
    match v {
        Value::Str(s) => py_repr_str(s),
        other => py_str(other),
    }
}

/// Python's `repr` of a string: single quotes unless that would need more
/// escaping than double quotes.
fn py_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse a JSON document.
///
/// Strict about the grammar, but two Python-isms are accepted because the corpus
/// is written by Python and the CLI is typed by a person: any top-level
/// whitespace, and integers of any width that fits `i128` (Python has no
/// overflow, so `17800000000000000000` is legal JSON for it).
pub fn parse(src: &str) -> Result<Value, String> {
    let b = src.as_bytes();
    let mut p = Parser { b, i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != b.len() {
        return Err(format!("trailing input at byte {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!(
                "expected `{}` at byte {}, found {}",
                c as char,
                self.i,
                match self.peek() {
                    Some(g) => format!("`{}`", g as char),
                    None => "end of input".to_string(),
                }
            ))
        }
    }

    fn lit(&mut self, s: &str) -> Result<(), String> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(())
        } else {
            Err(format!("expected `{s}` at byte {}", self.i))
        }
    }

    fn value(&mut self) -> Result<Value, String> {
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b't') => {
                self.lit("true")?;
                Ok(Value::Bool(true))
            }
            Some(b'f') => {
                self.lit("false")?;
                Ok(Value::Bool(false))
            }
            Some(b'n') => {
                self.lit("null")?;
                Ok(Value::Null)
            }
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(format!("unexpected `{}` at byte {}", c as char, self.i)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn object(&mut self) -> Result<Value, String> {
        self.eat(b'{')?;
        let mut kv = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Value::Object(kv));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            self.eat(b':')?;
            self.ws();
            let v = self.value()?;
            kv.push((k, v));
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Value::Object(kv));
                }
                _ => return Err(format!("expected `,` or `}}` at byte {}", self.i)),
            }
        }
    }

    fn array(&mut self) -> Result<Value, String> {
        self.eat(b'[')?;
        let mut a = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Value::Array(a));
        }
        loop {
            self.ws();
            a.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Value::Array(a));
                }
                _ => return Err(format!("expected `,` or `]` at byte {}", self.i)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let Some(c) = self.peek() else {
                return Err("unterminated string".to_string());
            };
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.i += 1;
                    let Some(e) = self.peek() else {
                        return Err("unterminated escape".to_string());
                    };
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            // A surrogate pair is two `\u` escapes; anything else
                            // is a lone surrogate and is not representable.
                            if (0xd800..0xdc00).contains(&hi) {
                                self.lit("\\u")?;
                                let lo = self.hex4()?;
                                if !(0xdc00..0xe000).contains(&lo) {
                                    return Err("a high surrogate not followed by a low one".to_string());
                                }
                                let cp =
                                    0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00);
                                out.push(
                                    char::from_u32(cp)
                                        .ok_or_else(|| "bad surrogate pair".to_string())?,
                                );
                            } else {
                                out.push(
                                    char::from_u32(hi)
                                        .ok_or_else(|| format!("\\u{hi:04x} is not a character"))?,
                                );
                            }
                        }
                        other => {
                            return Err(format!("unknown escape `\\{}`", other as char));
                        }
                    }
                }
                _ => {
                    // Copy one UTF-8 scalar; the input is a `&str`, so it is valid.
                    let start = self.i;
                    let len = utf8_len(c);
                    self.i += len;
                    out.push_str(
                        std::str::from_utf8(&self.b[start..self.i])
                            .map_err(|e| format!("invalid UTF-8: {e}"))?,
                    );
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() {
            return Err("truncated \\u escape".to_string());
        }
        let s = std::str::from_utf8(&self.b[self.i..self.i + 4])
            .map_err(|e| format!("invalid UTF-8 in \\u escape: {e}"))?;
        let v = u32::from_str_radix(s, 16).map_err(|_| format!("`{s}` is not four hex digits"))?;
        self.i += 4;
        Ok(v)
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        let int_start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
        let int_digits = &self.b[int_start..self.i];
        // JSON, like Python, forbids `01` and a bare `1.`.
        if int_digits.len() > 1 && int_digits[0] == b'0' {
            return Err(format!("a number may not have a leading zero, at byte {int_start}"));
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.i += 1;
            let frac_start = self.i;
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.i += 1;
            }
            if self.i == frac_start {
                return Err(format!("a `.` must be followed by a digit, at byte {frac_start}"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            let exp_start = self.i;
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.i += 1;
            }
            if self.i == exp_start {
                return Err(format!("an exponent must have digits, at byte {exp_start}"));
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i])
            .map_err(|e| format!("invalid UTF-8 in a number: {e}"))?;
        if int_digits.is_empty() && !is_float {
            return Err(format!("`{text}` is not a number"));
        }
        if is_float {
            text.parse::<f64>()
                .map(Value::Float)
                .map_err(|e| format!("`{text}` is not a float: {e}"))
        } else {
            text.parse::<i128>()
                .map(Value::Int)
                .map_err(|e| format!("`{text}` is not an i128: {e}"))
        }
    }
}

/// The byte length of a UTF-8 sequence from its leading byte. Only called with a
/// byte that starts one, since the input is a `&str`.
fn utf8_len(b: u8) -> usize {
    match b {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> Value {
        Value::Str(v.to_string())
    }

    fn obj(kv: &[(&str, Value)]) -> Value {
        Value::Object(kv.iter().map(|(k, v)| (k.to_string(), v.clone())).collect())
    }

    /// The exact strings `python3 -c 'import json; print(json.dumps(...))'`
    /// produced for these inputs. Every one of them is a case where `serde_json`
    /// would differ.
    #[test]
    fn dumps_matches_python() {
        assert_eq!(dumps(&obj(&[("a", Value::Int(1)), ("b", Value::Int(2))])), r#"{"a": 1, "b": 2}"#);
        assert_eq!(
            dumps(&obj(&[("a", obj(&[("b", Value::Array(vec![Value::Int(1), Value::Int(2), obj(&[("c", Value::Null)])]))])), ("d", Value::Bool(true)), ("e", Value::Array(vec![]))])),
            r#"{"a": {"b": [1, 2, {"c": null}]}, "d": true, "e": []}"#
        );
        assert_eq!(dumps(&Value::Object(vec![])), "{}");
        assert_eq!(dumps(&Value::Array(vec![])), "[]");
        // `ensure_ascii=False`, and no HTML-safe escaping.
        assert_eq!(dumps(&obj(&[("k", s("中文 é"))])), "{\"k\": \"中文 é\"}");
        assert_eq!(dumps(&obj(&[("k", s("<a>&'</a>"))])), "{\"k\": \"<a>&'</a>\"}");
        // Only `"`, `\` and the C0 controls are escaped.
        assert_eq!(dumps(&s("a\"b")), "\"a\\\"b\"");
        assert_eq!(dumps(&s("a\\b")), "\"a\\\\b\"");
        assert_eq!(dumps(&s("a\nb")), "\"a\\nb\"");
        assert_eq!(dumps(&s("a\tb")), "\"a\\tb\"");
        assert_eq!(dumps(&s("\u{0}\u{1}\u{1f}")), "\"\\u0000\\u0001\\u001f\"");
        assert_eq!(dumps(&s("\u{7f}")), "\"\u{7f}\"");
        assert_eq!(dumps(&s("\u{2028}\u{2029}")), "\"\u{2028}\u{2029}\"");
        assert_eq!(dumps(&s("\u{1F600}")), "\"\u{1F600}\"");
        assert_eq!(dumps(&s("a/b")), "\"a/b\"");
    }

    /// Key order is insertion order, which `serde_json`'s default `BTreeMap`
    /// would sort. A tool definition written `name, description, parameters` must
    /// not come back `description, name, parameters`.
    #[test]
    fn object_key_order_is_preserved() {
        let v = obj(&[("name", s("f")), ("description", s("d")), ("parameters", obj(&[]))]);
        assert_eq!(dumps(&v), r#"{"name": "f", "description": "d", "parameters": {}}"#);
        let back = parse(&dumps(&v)).unwrap();
        assert_eq!(back, v, "a round trip through the parser must not reorder");
    }

    /// The float cases, copied from `repr()` on CPython 3.10. The interesting
    /// ones are the boundaries: `1e-5` uses an exponent but `1e-4` does not, and
    /// `1e16` does but `1e15` does not.
    #[test]
    fn float_repr_matches_python() {
        let cases: &[(f64, &str)] = &[
            (1.5, "1.5"),
            (0.1, "0.1"),
            (1e-5, "1e-05"),
            (1e-4, "0.0001"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1e17, "1e+17"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (1.0 / 3.0, "0.3333333333333333"),
            (1e300, "1e+300"),
            (3.0, "3.0"),
            (100.0, "100.0"),
            // Written with the digits f64 can actually hold: the point is that a
            // value with more precision than f64 has still round-trips to the
            // shortest spelling, which is what Python prints.
            (123_456_789.123_456_79, "123456789.12345679"),
            (2.5e-323, "2.5e-323"),
            (f64::MAX, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
        ];
        for (v, want) in cases {
            assert_eq!(&py_repr_f64(*v), want, "repr({v:?})");
        }
    }

    /// `str` and `dumps` differ, and the template needs both: a bare string, int,
    /// bool or float goes through `str` (no quotes, `True`, `None`), while a
    /// mapping or a non-string sequence goes through `tojson`.
    #[test]
    fn py_str_is_pythons_str() {
        assert_eq!(py_str(&Value::Int(42)), "42");
        assert_eq!(py_str(&Value::Float(1.5)), "1.5");
        assert_eq!(py_str(&Value::Bool(true)), "True");
        assert_eq!(py_str(&Value::Bool(false)), "False");
        assert_eq!(py_str(&Value::Null), "None");
        assert_eq!(py_str(&s("s")), "s");
        assert_eq!(py_str(&Value::Array(vec![Value::Int(1), Value::Int(2)])), "[1, 2]");
        assert_eq!(py_str(&obj(&[("x", Value::Int(1))])), "{'x': 1}");
        assert_eq!(py_str(&Value::Array(vec![])), "[]");
        // The container branches are unreachable from the template, but a
        // container holding a string must still quote it, pythonically.
        assert_eq!(py_str(&Value::Array(vec![s("it's")])), "[\"it's\"]");
        assert_eq!(py_str(&Value::Array(vec![s("a\nb")])), "['a\\nb']");
    }

    #[test]
    fn parses_every_shape() {
        assert_eq!(parse("null").unwrap(), Value::Null);
        assert_eq!(parse(" true ").unwrap(), Value::Bool(true));
        assert_eq!(parse("-17").unwrap(), Value::Int(-17));
        assert_eq!(parse("1.5e3").unwrap(), Value::Float(1500.0));
        assert_eq!(parse(r#""a\u0041\u4e2d""#).unwrap(), s("aA中"));
        // A surrogate pair, and the non-BMP character it stands for.
        assert_eq!(parse(r#""\ud83d\ude00""#).unwrap(), s("\u{1F600}"));
        assert_eq!(parse("[]").unwrap(), Value::Array(vec![]));
        assert_eq!(parse("{}").unwrap(), Value::Object(vec![]));
        assert_eq!(
            parse(r#"{"a": [1, {"b": null}]}"#).unwrap(),
            obj(&[("a", Value::Array(vec![Value::Int(1), obj(&[("b", Value::Null)])]))])
        );
        // Python has arbitrary-precision ints, so an id beyond i64 is legal.
        assert_eq!(parse("17800000000000000000").unwrap(), Value::Int(17800000000000000000));
    }

    #[test]
    fn rejects_malformed_input() {
        for bad in [
            "", "{", "[1,", "{\"a\"}", "tru", "01", "-01", "1.", "1e", "\"unterminated", "1 2",
            "[1]]",
        ] {
            assert!(parse(bad).is_err(), "`{bad}` should not parse");
        }
        // A lone high surrogate has no character, so it is refused rather than
        // silently replaced.
        assert!(parse(r#""\ud83d""#).is_err());
        // ... but a leading zero inside a string is just a string.
        assert_eq!(parse(r#""01""#).unwrap(), s("01"));
    }

    /// `dumps` then `parse` must be the identity on every value, which is the
    /// property the corpus relies on when it round-trips tool definitions.
    #[test]
    fn dumps_parse_round_trip() {
        let v = obj(&[
            ("s", s("a\"b\\c\nd")),
            ("i", Value::Int(-3)),
            ("f", Value::Float(1e-5)),
            ("b", Value::Bool(false)),
            ("n", Value::Null),
            ("a", Value::Array(vec![Value::Float(0.0), s("")])),
            ("o", obj(&[("z", Value::Int(1)), ("a", Value::Int(2))])),
        ]);
        assert_eq!(parse(&dumps(&v)).unwrap(), v);
    }
}
