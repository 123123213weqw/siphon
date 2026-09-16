//! `qwenrun` — run a real Qwen3.5 checkpoint.
//!
//! ```text
//! qwenrun <model-dir> [--prompt 1,2,3] [--tokens N] [--dump-logits FILE]
//!                     [--compare FILE] [--topk K]
//! ```
//!
//! Prints what it loaded, runs the prompt through the stack, reports the next-token
//! distribution, and optionally decodes greedily.
//!
//! # Comparing against the reference
//!
//! `--dump-logits` writes the last position's logits as raw little-endian `f32`, and
//! `--compare` reads such a file and diffs it. The expected workflow is: run the
//! reference (transformers) once, dump the same tensor, then compare. A tolerance of
//! `1e-5` is used by default; a `bf16` checkpoint cannot be expected to agree much
//! more closely than its own storage precision, so the interesting number is whether
//! the **argmax** and the top-k ordering match, which the tool reports separately.

use std::process::ExitCode;

use gdn::chat;
use gdn::chatparse;
use gdn::model;
use gdn::pyjson;
use gdn::real;

/// One turn on the command line, in the order it was written.
///
/// The order matters: `--chat a --reply b --chat c` is a three-turn conversation,
/// and the chat template renders it differently from `--chat a --chat c --reply b`
/// (which is not a conversation at all). So the arguments are walked in order
/// rather than looked up by name.
enum Turn {
    System(String),
    User(String),
    Assistant(String),
    Tool(String),
}

fn parse_turns(args: &[String]) -> Result<Vec<Turn>, String> {
    let mut out = Vec::new();
    let mut i = 1usize;
    while i < args.len() {
        let a = args[i].as_str();
        let take = |i: usize, name: &str| -> Result<String, String> {
            args.get(i + 1).cloned().ok_or_else(|| format!("{name} needs a value"))
        };
        match a {
            "--system" => {
                out.push(Turn::System(take(i, a)?));
                i += 2;
            }
            "--chat" => {
                out.push(Turn::User(take(i, a)?));
                i += 2;
            }
            "--reply" => {
                out.push(Turn::Assistant(take(i, a)?));
                i += 2;
            }
            "--tool-result" => {
                out.push(Turn::Tool(take(i, a)?));
                i += 2;
            }
            _ => i += 1,
        }
    }
    Ok(out)
}

fn turns_to_messages(turns: &[Turn]) -> Vec<chat::Message> {
    turns
        .iter()
        .map(|t| match t {
            Turn::System(s) => chat::Message::system(s.clone()),
            Turn::User(s) => chat::Message::user(s.clone()),
            Turn::Assistant(s) => chat::Message::assistant(s.clone()),
            Turn::Tool(s) => chat::Message::tool(s.clone()),
        })
        .collect()
}

/// Show the calls a reply asked for. The values are text, because that is what a
/// `<parameter>` holds; mapping them onto the tool's schema is the caller's job.
fn print_tool_calls(reply: &chatparse::Reply) {
    for c in &reply.tool_calls {
        let args = c
            .arguments
            .iter()
            .map(|(k, v)| format!("{k}={v:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!("   tool call: {}({args})", c.name);
    }
}

/// The conversational path: render the template, decode until the turn ends, and
/// read the reply back.
///
/// One iteration per turn; `--repl` drives it in a loop.
#[allow(clippy::too_many_arguments)]
fn run_chat(
    c: &model::ModelConfig,
    w: &model::ModelWeights,
    tk: &gdn::tokenizer::Tokenizer,
    mut messages: Vec<chat::Message>,
    tools: Vec<pyjson::Value>,
    opts: chat::Options,
    steps: usize,
    repl: bool,
    show_prompt: bool,
) -> ExitCode {
    // `<|im_end|>` is the checkpoint's `eos_token` -- measured from
    // `tokenizer_config.json`, not assumed -- and the template closes every turn
    // with it. It is a stop token rather than part of the reply.
    let stop: Vec<u32> = vec![248046];
    let mut stdin_lines = std::io::stdin().lines();
    loop {
        // In a REPL the next turn is read *before* anything is rendered: rendering
        // first would have to invent a turn to render, and a greeting the user did
        // not type is a turn they did not ask for.
        if repl {
            print!("user> ");
            use std::io::Write as _;
            let _ = std::io::stdout().flush();
            let Some(Ok(line)) = stdin_lines.next() else {
                println!();
                return ExitCode::SUCCESS;
            };
            if line.trim().is_empty() {
                return ExitCode::SUCCESS;
            }
            messages.push(chat::Message::user(line));
        }

        let req = chat::Request {
            messages: messages.clone(),
            tools: tools.clone(),
            opts,
        };
        let rendered = match chat::render(&req) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("chat template: {e}");
                return ExitCode::FAILURE;
            }
        };
        println!();
        println!(
            "== chat prompt ({} bytes, {} turn(s))",
            rendered.text.len(),
            messages.len()
        );
        // The prompt is echoed in a one-shot run, where seeing it is the point, and
        // summarised in a REPL, where it is a wall of text on every turn.
        if repl {
            println!("   (use --show-prompt to echo it)");
        }
        if show_prompt || !repl {
            println!("   {:?}", rendered.text);
        }
        if rendered.images > 0 || rendered.videos > 0 {
            println!(
                "   !! {} image and {} video part(s) rendered; this engine loads only the \
                 language model, so those ids have no vision embedding behind them",
                rendered.images, rendered.videos
            );
        }
        let prompt = tk.encode(&rendered.text);
        println!("   -> {} tokens", prompt.len());

        // A fresh cache per turn: the whole conversation is re-rendered each turn
        // (the assistant block is closed and re-emitted as history), so the cached
        // prefix would not survive the re-render anyway.
        let t0 = std::time::Instant::now();
        let (generated, _, stopped) =
            match model::greedy_cached_stopping(c, w, &prompt, steps, &stop) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("decode failed: {e}");
                    return ExitCode::FAILURE;
                }
            };
        let secs = t0.elapsed().as_secs_f64();

        // The prompt ends inside the assistant turn, so everything generated is the
        // continuation -- which is exactly what `parse_assistant` reads.
        let continuation = tk.decode(&generated);
        let reply = chatparse::parse_assistant(&continuation);
        println!();
        println!(
            "   {} token(s) in {secs:.2}s ({:.2}s/token){}",
            generated.len(),
            secs / generated.len().max(1) as f64,
            if stopped { "" } else { "  [hit the step limit, not <|im_end|>]" }
        );
        println!("   generated ids: {}", generated.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
        if let Some(r) = &reply.reasoning {
            println!("   reasoning: {r:?}");
        }
        println!("   reply:     {:?}", reply.content);
        print_tool_calls(&reply);
        if !reply.tool_calls.is_empty() {
            println!(
                "   to answer a call, re-run with the assistant turn and the result:\n\
                 \x20    --reply {continuation:?} --tool-result OUTPUT"
            );
        }

        if !repl {
            return ExitCode::SUCCESS;
        }
        // The assistant's own turn goes into the history with its reasoning and its
        // calls, so the next render sees the conversation the model actually had.
        let mut assistant = chat::Message::assistant(reply.content.clone());
        assistant.reasoning = Some(reply.reasoning.clone().unwrap_or_default());
        assistant.tool_calls = reply.as_chat_calls();
        messages.push(assistant);
    }
}

/// Decode a raw little-endian `f32` blob.
///
/// `chunks(4)` rather than `chunks_exact(4)`: clippy flags the latter with a constant
/// chunk size, and the trailing-partial case cannot arise here anyway because the
/// caller rejects a length that is not a multiple of 4.
fn read_f32_blob(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks(4)
        .filter(|c| c.len() == 4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn parse_list(s: &str) -> Result<Vec<u32>, String> {
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| p.trim().parse::<u32>().map_err(|e| format!("bad token id `{p}`: {e}")))
        .collect()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let val = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    let Some(dir) = args.iter().skip(1).find(|a| !a.starts_with("--")) else {
        eprintln!(
            "usage: qwenrun <model-dir> [--text STRING | --prompt 1,2,3 | --chat STRING ...]\n\
             \x20                    [--tokens N] [--topk K] [--cached] [--expect-tokens IDS]\n\
             \x20                    [--dump-logits FILE] [--compare FILE]\n\
             \x20       chat: --system S --chat S --reply S --tool-result S --thinking --repl\n\
             \x20             --tools-file FILE.json | --tools JSON"
        );
        return ExitCode::from(2);
    };
    let turns = match parse_turns(&args) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let repl = args.iter().any(|a| a == "--repl");
    let chat_mode = repl || !turns.is_empty();
    let topk: usize = val("--topk").and_then(|s| s.parse().ok()).unwrap_or(10);
    let steps: usize = val("--tokens").and_then(|s| s.parse().ok()).unwrap_or(0);
    // Default tolerance, and why it is not 1e-5.
    //
    // A `bf16` checkpoint loaded into an `f32` reference is still an `f32`
    // computation, and `f32` does not reproduce itself across implementations. On
    // this model the reference disagrees with *itself* by 2.46e-5 between CPU and
    // GPU (same weights, same dtype, same eager attention), and sits 1.6e-5 to
    // 2.6e-5 from a float64 ground truth.
    //
    // So a bound tighter than that is not a test of correctness; it is a demand that
    // this engine agree with the reference more closely than the reference agrees
    // with itself. 3e-5 is above the reference's own spread and below what a real
    // operator mistake produces (a wrong gate or a missing rotation moves logits by
    // ~1e-1, not 1e-5).
    let tol: f32 = val("--tol").and_then(|s| s.parse().ok()).unwrap_or(3e-5);
    let use_cache = args.iter().any(|a| a == "--cached");
    // A text prompt goes through the tokenizer; a numeric one is used as-is, which is
    // what the reference comparison scripts exchange.
    let text_prompt = val("--text");
    let (prompt, tokenizer) = match (&text_prompt, val("--prompt")) {
        (Some(t), _) => {
            let (tk, info) = match gdn::tokenizer::Tokenizer::from_model_dir(dir) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let ids = tk.encode(t);
            println!("== tokenizer");
            println!(
                "   {}  vocab {}  merges {}  added {}  pattern {}",
                info.model_type, info.vocab_size, info.merges, info.added_tokens,
                info.pattern_variant
            );
            println!("   prompt {t:?}");
            println!("   -> {} tokens {ids:?}", ids.len());
            println!();
            (ids, Some(tk))
        }
        (None, Some(s)) => match parse_list(&s) {
            Ok(v) => (v, None),
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        },
        (None, None) => (vec![9419], None),
    };
    if prompt.is_empty() {
        eprintln!("error: empty prompt");
        return ExitCode::from(2);
    }

    // ---- load -------------------------------------------------------------
    println!("== loading {dir}");
    let t0 = std::time::Instant::now();
    let rm = match real::load(dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let load_s = t0.elapsed().as_secs_f64();
    let i = &rm.info;
    println!("   loaded in {load_s:.1}s");
    println!(
        "   prefix `{}`  config from {}  {} shard(s)  {} tensors  {:.2} GiB of tensor data",
        i.prefix,
        i.config_source,
        i.shards,
        i.tensors_total,
        i.tensor_bytes as f64 / (1u64 << 30) as f64
    );
    println!(
        "   dtypes: {}",
        i.dtype_counts
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    if !i.skipped.is_empty() {
        println!(
            "   unused sub-trees: {}",
            i.skipped
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    println!(
        "   head: {}",
        if i.tied_embeddings {
            "tied to embed_tokens".to_string()
        } else {
            "separate lm_head".to_string()
        }
    );
    print!("   derived:");
    for (k, v) in &i.derived {
        print!(" {k}={v}");
    }
    println!();
    let c = &rm.config;
    println!(
        "   gdn: k_heads={} v_heads={} head_k={} head_v={} conv_k={}",
        c.gdn.num_k_heads, c.gdn.num_v_heads, c.gdn.head_k_dim, c.gdn.head_v_dim, c.gdn.conv_kernel
    );
    println!(
        "   attn: heads={} kv_heads={} head_dim={} rotary_dim={} theta={}",
        c.attn.num_heads, c.attn.num_kv_heads, c.attn.head_dim, c.attn.rotary_dim, c.attn.rope_theta
    );
    let lin = rm
        .weights
        .layers
        .iter()
        .filter(|l| l.kind == model::LayerKind::LinearAttention)
        .count();
    println!(
        "   {} layers ({} linear_attention, {} full_attention), hidden={} vocab={} eps={:e}",
        rm.weights.layers.len(),
        lin,
        rm.weights.layers.len() - lin,
        c.hidden,
        c.vocab,
        c.eps
    );

    // ---- chat --------------------------------------------------------------
    //
    // Placed here so it shares the loading report above and skips the raw
    // continuation path below, which is a different question ("what comes next"
    // rather than "what is the answer").
    if chat_mode {
        let (tk, info) = match gdn::tokenizer::Tokenizer::from_model_dir(dir) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        let tools: Vec<pyjson::Value> = if let Some(path) = val("--tools-file") {
            match std::fs::read(&path).map_err(|e| format!("{path}: {e}")) {
                Ok(b) => match pyjson::parse(&String::from_utf8_lossy(&b)) {
                    Ok(pyjson::Value::Array(a)) => a,
                    Ok(one @ pyjson::Value::Object(_)) => vec![one],
                    Ok(_) => {
                        eprintln!("{path}: tools must be an array or an object");
                        return ExitCode::FAILURE;
                    }
                    Err(e) => {
                        eprintln!("{path}: {e}");
                        return ExitCode::FAILURE;
                    }
                },
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
            }
        } else if let Some(inline) = val("--tools") {
            match pyjson::parse(&inline) {
                Ok(pyjson::Value::Array(a)) => a,
                Ok(one @ pyjson::Value::Object(_)) => vec![one],
                Ok(_) => {
                    eprintln!("--tools must be an array or an object");
                    return ExitCode::FAILURE;
                }
                Err(e) => {
                    eprintln!("error: --tools: {e}");
                    return ExitCode::FAILURE;
                }
            }
        } else {
            Vec::new()
        };
        let opts = chat::Options {
            add_generation_prompt: true,
            enable_thinking: args.iter().any(|a| a == "--thinking"),
            add_vision_id: args.iter().any(|a| a == "--vision-id"),
        };
        println!();
        println!(
            "== chat  {}  added {}  pattern {}  {} tool(s)  thinking {}",
            info.model_type,
            info.added_tokens,
            info.pattern_variant,
            tools.len(),
            opts.enable_thinking
        );
        // In chat mode the turn list is the prompt; in a REPL it is read from stdin.
        // At least one user turn is required, and the template says so if there is
        // not one -- but a REPL with only a `--system` is legal, so it is not
        // rejected here.
        let steps = if steps == 0 { 64 } else { steps };
        let messages = turns_to_messages(&turns);
        return run_chat(
            c,
            &rm.weights,
            &tk,
            messages,
            tools,
            opts,
            steps,
            repl,
            args.iter().any(|a| a == "--show-prompt"),
        );
    }

    // ---- forward ----------------------------------------------------------
    println!();
    println!("   prompt: {prompt:?}  ({} tokens)", prompt.len());
    let t1 = std::time::Instant::now();
    let tr = match model::forward(c, &rm.weights, &prompt) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("forward failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let fwd_s = t1.elapsed().as_secs_f64();
    println!(
        "   forward: {fwd_s:.2}s for {} token(s)  ({:.3}s/token)",
        prompt.len(),
        fwd_s / prompt.len() as f64
    );

    let last = tr.last_logits();
    if !last.iter().all(|x| x.is_finite()) {
        let bad = last.iter().filter(|x| !x.is_finite()).count();
        eprintln!("   !! {bad} non-finite logits");
        return ExitCode::FAILURE;
    }
    let mut idx: Vec<usize> = (0..last.len()).collect();
    idx.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
    println!("   next-token argmax = {}", idx[0]);
    println!("   top-{topk}:");
    for &t in idx.iter().take(topk) {
        match &tokenizer {
            Some(tk) => println!(
                "      {:>7}  {:+.6}  {:?}",
                t,
                last[t],
                tk.decode(&[t as u32])
            ),
            None => println!("      {:>7}  {:+.6}", t, last[t]),
        }
    }
    // A distribution that is nearly uniform would mean the stack is not doing
    // anything; a healthy model puts most of its mass on a few tokens.
    let max = last[idx[0]];
    let sum: f64 = last.iter().map(|&x| ((x - max) as f64).exp()).sum();
    let p0 = 1.0 / sum;
    println!("   p(argmax) = {p0:.4}   (uniform would be {:.2e})", 1.0 / c.vocab as f64);

    let mut failed = false;

    // ---- per-layer comparison ---------------------------------------------
    // The point of this: an end-to-end logit diff of 1e-4 tells you nothing about
    // whether it is 24 layers of slow accumulation or one layer that is wrong. The
    // reference dumps every layer output, so each layer can be judged on its own.
    if let Some(dir) = val("--compare-layers") {
        let dir = std::path::PathBuf::from(dir);
        let shapes_path = dir.join("layer_shapes.json");
        let blob_path = dir.join("layers.f32");
        let shapes: Vec<Vec<usize>> = match std::fs::read(&shapes_path)
            .map_err(|e| format!("{}: {e}", shapes_path.display()))
            .and_then(|b| {
                serde_json::from_slice(&b).map_err(|e| format!("{}: {e}", shapes_path.display()))
            }) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        let blob = match std::fs::read(&blob_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: {}: {e}", blob_path.display());
                return ExitCode::FAILURE;
            }
        };
        let refv = read_f32_blob(&blob);
        let expect: usize = shapes.iter().map(|s| s.iter().product::<usize>()).sum();
        if refv.len() != expect {
            eprintln!(
                "{}: {} values but the shapes need {expect}",
                blob_path.display(),
                refv.len()
            );
            return ExitCode::FAILURE;
        }
        println!();
        println!("   per-layer agreement (each layer's output, mine vs reference)");
        println!("     {:<6} {:>12} {:>12} {:>12}", "layer", "max abs", "rel", "growth");
        let mut off = 0usize;
        let mut prev = 0f32;
        for (k, sh) in shapes.iter().enumerate() {
            let n: usize = sh.iter().product();
            let gold = &refv[off..off + n];
            off += n;
            if k >= tr.layers.len() {
                println!("     {k:<6} (mine has only {} layers)", tr.layers.len());
                break;
            }
            let mine = &tr.layers[k].out;
            if mine.len() != n {
                println!("     {k:<6} SHAPE mine={} ref={n}", mine.len());
                failed = true;
                continue;
            }
            let mut worst = 0f32;
            for j in 0..n {
                let d = (mine[j] - gold[j]).abs();
                if d > worst {
                    worst = d;
                }
            }
            let scale = gold.iter().fold(0f32, |m, x| m.max(x.abs()));
            let rel = if scale > 0.0 { worst / scale } else { worst };
            println!(
                "     {k:<6} {worst:>12.3e} {rel:>12.3e} {:>12}",
                if prev > 0.0 {
                    format!("{:.1}x", worst / prev)
                } else {
                    "-".to_string()
                }
            );
            prev = worst;
            if worst > tol {
                failed = true;
            }
        }
        println!(
            "     note: growth is the ratio to the previous layer, so a gradual rise is \
             accumulation and a jump is a layer that is wrong"
        );
    }

    // ---- dump / compare ----------------------------------------------------
    if let Some(path) = val("--dump-logits") {
        let mut bytes = Vec::with_capacity(last.len() * 4);
        for v in last {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        match std::fs::write(&path, &bytes) {
            Ok(()) => println!("   dumped {} f32 logits to {path}", last.len()),
            Err(e) => {
                eprintln!("cannot write {path}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    if let Some(path) = val("--compare") {
        match std::fs::read(&path) {
            Ok(bytes) => {
                if bytes.len() % 4 != 0 {
                    eprintln!("{path}: {} bytes is not a multiple of 4", bytes.len());
                    return ExitCode::FAILURE;
                }
                let refv = read_f32_blob(&bytes);
                if refv.len() != last.len() {
                    eprintln!(
                        "{path}: {} values but the model has {} logits",
                        refv.len(),
                        last.len()
                    );
                    return ExitCode::FAILURE;
                }
                let (mut worst, mut at) = (0f32, 0usize);
                for k in 0..refv.len() {
                    let d = (last[k] - refv[k]).abs();
                    if d > worst {
                        worst = d;
                        at = k;
                    }
                }
                let mut ridx: Vec<usize> = (0..refv.len()).collect();
                ridx.sort_by(|&a, &b| refv[b].partial_cmp(&refv[a]).unwrap());
                let my_top: Vec<usize> = idx.iter().take(topk).copied().collect();
                let ref_top: Vec<usize> = ridx.iter().take(topk).copied().collect();
                println!();
                println!("   compare vs {path}");
                println!("     max abs diff      {worst:.3e}  (tolerance {tol:.0e})");
                println!("     worst at index    {at}  mine={} ref={}", last[at], refv[at]);
                println!("     argmax            mine={} ref={}", idx[0], ridx[0]);
                println!(
                    "     top-{topk} ordering  {}",
                    if my_top == ref_top { "identical" } else { "DIFFER" }
                );
                if my_top != ref_top {
                    println!("       mine {my_top:?}");
                    println!("       ref  {ref_top:?}");
                }
                // The tokens are what matters; the absolute tolerance is a proxy.
                // Two independent gates, reported separately: the functional one
                // (do the tokens agree) and the numerical one (are the values within
                // the tolerance). They fail for different reasons and the distinction
                // matters -- identical tokens with a large value error means a
                // precision problem, whereas differing tokens means a wrong operator.
                let ok = idx[0] == ridx[0] && my_top == ref_top;
                if !ok || worst > tol {
                    failed = true;
                }
                println!(
                    "     => {}",
                    if !ok {
                        "FAIL: the tokens differ, which is an operator error rather than precision"
                    } else if worst <= tol {
                        "PASS"
                    } else {
                        "FAIL: tokens agree but the values exceed the tolerance"
                    }
                );
            }
            Err(e) => {
                eprintln!("cannot read {path}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    // ---- greedy -----------------------------------------------------------
    if steps > 0 {
        println!();
        println!(
            "   greedy decoding {steps} token(s){}",
            if use_cache { " [cached: prefill once, then one token per step]" } else { " [no cache: whole prefix re-run each step]" }
        );
        let mut ids = prompt.clone();
        let t2 = std::time::Instant::now();

        // Collect the argmax per step first, so the two modes print identically and the
        // only thing that differs is how the logits were obtained.
        let mut produced: Vec<u32> = Vec::with_capacity(steps);
        let mut cache_note = String::new();

        if use_cache {
            let mut cache = match model::Cache::new(c, &rm.weights, 1) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("cannot allocate cache: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let (lin0, full0) = cache.bytes_by_kind();
            let (n_lin, n_full) = rm
                .weights
                .layers
                .iter()
                .fold((0usize, 0usize), |(l, f), w| match w.kind {
                    model::LayerKind::LinearAttention => (l + 1, f),
                    model::LayerKind::FullAttention => (l, f + 1),
                });
            let per_token = n_full * 2 * c.attn.num_kv_heads * c.attn.head_dim * 4;
            println!(
                "     cache at len 0: linear {:.2} MiB across {} layers (constant) + \
                 full {:.2} MiB across {} layers, growing by {:.1} KiB/token",
                lin0 as f64 / 1048576.0,
                n_lin,
                full0 as f64 / 1048576.0,
                n_full,
                per_token as f64 / 1024.0
            );
            // Prefill the prompt in one call, then one token at a time.
            let mut tr = match model::forward_cached(c, &rm.weights, &mut cache, &prompt) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("prefill failed: {e}");
                    return ExitCode::FAILURE;
                }
            };
            for k in 0..steps {
                let nxt = tr.argmax_last() as u32;
                ids.push(nxt);
                produced.push(nxt);
                match &tokenizer {
                    Some(tk) => println!(
                        "     step {k:>2}  len={:<4} -> {:>7}  {:?}   ({:.2}s elapsed)",
                        ids.len() - 1,
                        nxt,
                        tk.decode(&[nxt]),
                        t2.elapsed().as_secs_f64()
                    ),
                    None => println!(
                        "     step {k:>2}  len={:<4} -> {nxt}   ({:.2}s elapsed)",
                        ids.len() - 1,
                        t2.elapsed().as_secs_f64()
                    ),
                }
                tr = match model::forward_cached(c, &rm.weights, &mut cache, &[nxt]) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("decode failed at step {k}: {e}");
                        return ExitCode::FAILURE;
                    }
                };
            }
            let (lin, full) = cache.bytes_by_kind();
            cache_note = format!(
                "cache after {} tokens: linear {:.2} MiB (unchanged) + full {:.2} MiB",
                cache.len,
                lin as f64 / 1048576.0,
                full as f64 / 1048576.0
            );
        } else {
            for k in 0..steps {
                let t = match model::forward(c, &rm.weights, &ids) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("forward failed at step {k}: {e}");
                        return ExitCode::FAILURE;
                    }
                };
                let nxt = t.argmax_last() as u32;
                ids.push(nxt);
                produced.push(nxt);
                match &tokenizer {
                    Some(tk) => println!(
                        "     step {k:>2}  len={:<4} -> {:>7}  {:?}   ({:.2}s elapsed)",
                        ids.len() - 1,
                        nxt,
                        tk.decode(&[nxt]),
                        t2.elapsed().as_secs_f64()
                    ),
                    None => println!(
                        "     step {k:>2}  len={:<4} -> {nxt}   ({:.2}s elapsed)",
                        ids.len() - 1,
                        t2.elapsed().as_secs_f64()
                    ),
                }
            }
        }
        if !cache_note.is_empty() {
            println!("   {cache_note}");
        }
        println!(
            "   generated ids: {}",
            produced.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
        );
        if let Some(tk) = &tokenizer {
            println!("   generated text: {:?}", tk.decode(&produced));
            println!("   full text:      {:?}", tk.decode(&ids));
        }

        // An expected sequence turns "does caching change the answer" into a check the
        // tool can fail, which matters because a wrong cache still produces fluent text.
        if let Some(exp) = val("--expect-tokens") {
            let want: Vec<u32> = match parse_list(&exp) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: --expect-tokens: {e}");
                    return ExitCode::from(2);
                }
            };
            let ok = want == produced;
            println!(
                "   expected {} token(s): {}",
                want.len(),
                if ok { "identical".to_string() } else { "DIFFER".to_string() }
            );
            if !ok {
                println!("     want {want:?}");
                println!("     got  {produced:?}");
                failed = true;
            }
        }
    }


    println!();
    if failed {
        println!("   RESULT: FAIL");
        ExitCode::FAILURE
    } else {
        println!("   RESULT: OK");
        ExitCode::SUCCESS
    }
}
