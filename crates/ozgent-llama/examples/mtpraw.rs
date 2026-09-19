//! MTP speculation on raw llama.cpp, beneath the engine.
//!
//! Where `examples/mtp.rs` measures what ozgent does, this measures what
//! llama.cpp will allow, and answers the question the engine cannot: does a
//! rejected draft leave the recurrent state wrong?
//!
//! Usage: `mtpraw <model.gguf> [draft_max] [tokens]`, plus:
//!
//! * `STATECHECK=1` — decode the same tokens twice, once plainly and once
//!   with a deliberately wrong draft every round so every round ends in a
//!   ring rollback, and compare the logits position by position. Both runs
//!   decode identical tokens, so any difference is the state itself rather
//!   than two answers drifting apart. Measured: exactly zero over 200 rounds
//!   after a 1,640-token prompt and over 300 after a 104-token one; after a
//!   52-token prompt, zero for 202 rounds and then a flat 0.2 of logits
//!   whose largest is 28, which changed one token in 300.
//! * `STATECHECK=1 PAIRS=1` — the control: the same tokens through two-token
//!   batches with no drafting and no rollback. Zero throughout, which is what
//!   makes the step above the ring's and not the batch shape's.
//! * `ROLLTEST=1` — the ring's edges. A rollback after a multi-token batch is
//!   exact; one of two tokens with a ring of one is refused; and a rollback
//!   after a *single-token* decode is accepted and silently wrong, which is
//!   why only draft verification ever trims a hybrid model's cache.
//! * `ORACLE=good|bad` — drafts taken from the undrafted run, right or
//!   deliberately wrong, to separate the drafter's quality from the
//!   machinery around it.
//! * `NOCATCHUP=1` — leave the head's own cache empty, to price catching it
//!   up. `PMIN`, `NGL`, `REPEAT`, `DUB`, `BSAMP`, `RS`, `NCTX`, `NOFA` and
//!   `PROMPT_FILE` tune the rest.

use ozgent_mtmd_sys::llama_cpp_sys_2 as sys;
use std::time::Instant;

unsafe extern "C" {
    #[link_name = "_Z26llama_set_embeddings_nextnP13llama_contextbb"]
    fn set_embeddings_nextn(ctx: *mut sys::llama_context, value: bool, masked: bool);
    #[link_name = "_Z30llama_get_embeddings_nextn_ithP13llama_contexti"]
    fn get_embeddings_nextn_ith(ctx: *mut sys::llama_context, i: i32) -> *mut f32;
    fn malloc(n: usize) -> *mut std::ffi::c_void;
}

const PROMPT: &str = "<|im_start|>user\nExplain in detail how a refrigerator works, covering the \
refrigerant cycle, the compressor, the condenser, the expansion valve and the evaporator, and why \
the back of the fridge gets warm.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

struct Batch {
    raw: sys::llama_batch,
    n_embd: usize,
}

impl Batch {
    fn new(cap: usize, n_embd: usize) -> Self {
        let mut raw = unsafe { sys::llama_batch_init(cap as i32, n_embd as i32, 1) };
        if n_embd > 0 {
            raw.token = unsafe { malloc(4 * cap) as *mut sys::llama_token };
        }
        Batch { raw, n_embd }
    }
    fn clear(&mut self) {
        self.raw.n_tokens = 0;
    }
    fn add(&mut self, tok: i32, pos: i32, logits: bool, h: Option<&[f32]>) {
        let i = self.raw.n_tokens as usize;
        unsafe {
            *self.raw.token.add(i) = tok;
            *self.raw.pos.add(i) = pos;
            *self.raw.n_seq_id.add(i) = 1;
            *(*self.raw.seq_id.add(i)).add(0) = 0;
            *self.raw.logits.add(i) = logits as i8;
            if self.n_embd > 0 {
                let dst = self.raw.embd.add(i * self.n_embd);
                match h {
                    Some(h) => std::ptr::copy_nonoverlapping(h.as_ptr(), dst, self.n_embd),
                    None => std::ptr::write_bytes(dst, 0, self.n_embd),
                }
            }
        }
        self.raw.n_tokens += 1;
    }
}

fn argmax(ctx: *mut sys::llama_context, i: i32, n_vocab: usize) -> (i32, f32) {
    let row = unsafe { std::slice::from_raw_parts(sys::llama_get_logits_ith(ctx, i), n_vocab) };
    let mut best = 0;
    for (j, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = j;
        }
    }
    let top = row[best];
    let sum: f32 = row.iter().map(|v| (v - top).exp()).sum();
    (best as i32, 1.0 / sum)
}

fn hrow(ctx: *mut sys::llama_context, i: i32, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(get_embeddings_nextn_ith(ctx, i), n).to_vec() }
}

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let k: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(3);
    let want: usize = std::env::args().nth(3).and_then(|v| v.parse().ok()).unwrap_or(300);
    let p_min: f32 = std::env::var("PMIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let ngl: i32 = std::env::var("NGL").ok().and_then(|v| v.parse().ok()).unwrap_or(999);
    let prompt_repeat: usize = std::env::var("REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);

    unsafe {
        sys::llama_backend_init();
        let mut mp = sys::llama_model_default_params();
        mp.n_gpu_layers = ngl;
        mp.load_mtp = true;
        let cpath = std::ffi::CString::new(path).unwrap();
        let model = sys::llama_model_load_from_file(cpath.as_ptr(), mp);
        assert!(!model.is_null());
        let vocab = sys::llama_model_get_vocab(model);
        let n_vocab = sys::llama_vocab_n_tokens(vocab) as usize;
        let n_embd = sys::llama_model_n_embd(model) as usize;

        let base_prompt = std::env::var("PROMPT_FILE").ok().map(|f| std::fs::read_to_string(f).unwrap()).unwrap_or_else(|| PROMPT.to_string());
        let text = base_prompt.repeat(prompt_repeat);
        let ctext = std::ffi::CString::new(text.clone()).unwrap();
        let mut toks = vec![0i32; text.len() + 16];
        let n = sys::llama_tokenize(vocab, ctext.as_ptr(), text.len() as i32, toks.as_mut_ptr(), toks.len() as i32, true, true);
        toks.truncate(n as usize);
        println!("prompt {} tokens, n_vocab {n_vocab}, n_embd {n_embd}, k {k}, p_min {p_min}", toks.len());

        let make_target = |rs: u32| {
            let mut cp = sys::llama_context_default_params();
            cp.n_ctx = 16384;
            cp.n_batch = 2048;
            cp.n_ubatch = 512;
            cp.n_seq_max = 1;
            cp.n_rs_seq = rs;
            if let Some(n) = std::env::var("NOUT").ok().and_then(|v| v.parse().ok()) {
                cp.n_outputs_max = n;
            }
            cp.type_k = sys::GGML_TYPE_Q8_0;
            cp.type_v = sys::GGML_TYPE_Q8_0;
            cp.flash_attn_type = if std::env::var("NOFA").is_ok() {
                sys::LLAMA_FLASH_ATTN_TYPE_DISABLED
            } else {
                sys::LLAMA_FLASH_ATTN_TYPE_ENABLED
            };
            cp.n_ctx = std::env::var("NCTX").ok().and_then(|v| v.parse().ok()).unwrap_or(16384);
            let c = sys::llama_init_from_model(model, cp);
            assert!(!c.is_null());
            c
        };

        let prefill = |ctx: *mut sys::llama_context, drafter: Option<(*mut sys::llama_context, &mut Vec<f32>)>| -> i32 {
            let mut b = Batch::new(2048, 0);
            let mut db = Batch::new(2048, n_embd);
            let mut drafter = drafter;
            let mut last = 0;
            for (ci, chunk) in toks.chunks(2048).enumerate() {
                b.clear();
                let base = ci * 2048;
                for (j, t) in chunk.iter().enumerate() {
                    b.add(*t, (base + j) as i32, base + j + 1 == toks.len(), None);
                }
                assert_eq!(sys::llama_decode(ctx, b.raw), 0);
                if let Some((dctx, pending)) = drafter.as_mut() {
                    db.clear();
                    for (j, t) in chunk.iter().enumerate() {
                        let h = if j == 0 { pending.clone() } else { hrow(ctx, j as i32 - 1, n_embd) };
                        db.add(*t, (base + j) as i32, false, Some(&h));
                    }
                    **pending = hrow(ctx, chunk.len() as i32 - 1, n_embd);
                    if std::env::var("NOCATCHUP").is_err() {
                        assert_eq!(sys::llama_decode(*dctx, db.raw), 0);
                    }
                }
                if base + chunk.len() == toks.len() {
                    last = argmax(ctx, chunk.len() as i32 - 1, n_vocab).0;
                }
            }
            last
        };

        if std::env::var("STATECHECK").is_ok() {
            // Does a rejected draft leave the recurrent state wrong?
            //
            // Both runs decode exactly the same tokens, so the two states must
            // stay equal. The drafted run is forced to reject every round: it
            // verifies [next, deliberately-wrong] and rolls the wrong one back
            // through the ring. Comparing the logits of each position against
            // the undrafted run then measures the state itself rather than two
            // answers drifting apart.
            let ctx = make_target(0);
            let mut b = Batch::new(2048, 0);
            let mut plain = prefill(ctx, None);
            let mut reference = Vec::new();
            let mut tokens = Vec::new();
            let mut pos = toks.len() as i32;
            for _ in 0..want {
                tokens.push(plain);
                b.clear();
                b.add(plain, pos, true, None);
                assert_eq!(sys::llama_decode(ctx, b.raw), 0);
                reference.push(std::slice::from_raw_parts(sys::llama_get_logits_ith(ctx, 0), n_vocab).to_vec());
                plain = argmax(ctx, 0, n_vocab).0;
                pos += 1;
            }
            sys::llama_free(ctx);

            // Control: the same tokens through two-token batches, with no
            // drafting and no rollback at all. Whatever this drifts by is the
            // batch shape, not the ring.
            if std::env::var("PAIRS").is_ok() {
                let ctx = make_target(0);
                let _ = prefill(ctx, None);
                let mut pos = toks.len() as i32;
                let (mut worst, mut worst_at, mut flips) = (0f32, 0usize, 0usize);
                let mut i = 0;
                while i + 1 < tokens.len() {
                    b.clear();
                    b.add(tokens[i], pos, true, None);
                    b.add(tokens[i + 1], pos + 1, true, None);
                    assert_eq!(sys::llama_decode(ctx, b.raw), 0);
                    for j in 0..2 {
                        let row = std::slice::from_raw_parts(sys::llama_get_logits_ith(ctx, j as i32), n_vocab);
                        let d = reference[i + j].iter().zip(row).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                        if d > worst {
                            worst = d;
                            worst_at = i + j;
                        }
                        let mut best = 0;
                        for (k, v) in row.iter().enumerate() {
                            if *v > row[best] { best = k; }
                        }
                        let r = &reference[i + j];
                        let mut theirs = 0;
                        for (k, v) in r.iter().enumerate() {
                            if *v > r[theirs] { theirs = k; }
                        }
                        if best != theirs {
                            flips += 1;
                        }
                    }
                    pos += 2;
                    i += 2;
                }
                println!(
                    "control, same tokens two at a time, no drafting: worst logit difference {worst:.5} \
                     (at round {worst_at}), different top token in {flips} of {} rounds",
                    tokens.len()
                );
                return;
            }

            let ctx = make_target(std::env::var("RS").ok().and_then(|v| v.parse().ok()).unwrap_or(1));
            let _ = prefill(ctx, None);
            let mut pos = toks.len() as i32;
            let (mut worst, mut worst_at, mut flips) = (0f32, 0usize, 0usize);
            let mut trail: Vec<(usize, f32)> = Vec::new();
            for (i, token) in tokens.iter().enumerate() {
                // The draft is always wrong, so every round ends in a rollback.
                let wrong = (tokens[(i + 7) % tokens.len()] + 1) % n_vocab as i32;
                b.clear();
                b.add(*token, pos, true, None);
                b.add(wrong, pos + 1, true, None);
                assert_eq!(sys::llama_decode(ctx, b.raw), 0);
                let row = std::slice::from_raw_parts(sys::llama_get_logits_ith(ctx, 0), n_vocab);
                let d = reference[i].iter().zip(row).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                if d > worst {
                    worst = d;
                    worst_at = i;
                }
                let mine = argmax(ctx, 0, n_vocab).0;
                let theirs = {
                    let r = &reference[i];
                    let mut best = 0;
                    for (j, v) in r.iter().enumerate() {
                        if *v > r[best] { best = j; }
                    }
                    best as i32
                };
                if mine != theirs {
                    flips += 1;
                }
                if i % 25 == 0 || d > 0.001 {
                    trail.push((i, d));
                }
                let ok = sys::llama_memory_seq_rm(sys::llama_get_memory(ctx), 0, pos + 1, -1);
                assert!(ok, "ring rollback refused at {i}");
                pos += 1;
            }
            println!(
                "{want} rounds, every draft rejected: worst logit difference {worst:.5} (at round {worst_at}),                  different top token in {flips} of {want} rounds"
            );
            let shown: Vec<String> = trail.iter().take(40).map(|(i, d)| format!("{i}:{d:.4}")).collect();
            println!("differences by round: {}", shown.join(" "));
            let scale = reference[worst_at].iter().fold(0f32, |m, v| m.max(v.abs()));
            println!("for scale, the largest logit in that row is {scale:.2}");
            return;
        }

        if std::env::var("ROLLTEST").is_ok() {
            // Is a 1-token ring rollback exact after (a) a prefill chunk, (b) single-token decodes?
            let ctx = make_target(1);
            let mut b = Batch::new(2048, 0);
            let n = toks.len();
            let logits_after = |ctx, b: &mut Batch, t: i32, p: i32| -> Vec<f32> {
                b.clear();
                b.add(t, p, true, None);
                assert_eq!(sys::llama_decode(ctx, b.raw), 0);
                std::slice::from_raw_parts(sys::llama_get_logits_ith(ctx, 0), n_vocab).to_vec()
            };
            // (a) prefill all but last, then decode last -> reference
            b.clear();
            for (j, t) in toks[..n - 1].iter().enumerate() { b.add(*t, j as i32, j == n - 2, None); }
            assert_eq!(sys::llama_decode(ctx, b.raw), 0);
            let r1 = logits_after(ctx, &mut b, toks[n - 1], (n - 1) as i32);
            // roll back the last token (single decode) and decode it again
            let ok = sys::llama_memory_seq_rm(sys::llama_get_memory(ctx), 0, (n - 1) as i32, -1);
            let r2 = logits_after(ctx, &mut b, toks[n - 1], (n - 1) as i32);
            let d = r1.iter().zip(&r2).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            println!("single-decode rollback: accepted={ok}, max logit diff {d}");
            // (b) decode two singles X,Y then roll back 1 and redo Y
            let x = argmax(ctx, 0, n_vocab).0;
            let _ = logits_after(ctx, &mut b, x, n as i32);
            let y = 13;
            let ry = logits_after(ctx, &mut b, y, n as i32 + 1);
            let ok = sys::llama_memory_seq_rm(sys::llama_get_memory(ctx), 0, n as i32 + 1, -1);
            let ry2 = logits_after(ctx, &mut b, y, n as i32 + 1);
            let d = ry.iter().zip(&ry2).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            println!("second single rollback: accepted={ok}, max logit diff {d}");
            // (c) roll back 2 (beyond ring) must be refused
            let ok = sys::llama_memory_seq_rm(sys::llama_get_memory(ctx), 0, n as i32, -1);
            println!("rollback of 2 with ring 1: accepted={ok}");
            // (d) rollback after a prefill chunk of many tokens: prefill fresh seq then drop last
            sys::llama_memory_clear(sys::llama_get_memory(ctx), true);
            b.clear();
            for (j, t) in toks.iter().enumerate() { b.add(*t, j as i32, false, None); }
            assert_eq!(sys::llama_decode(ctx, b.raw), 0);
            let ok = sys::llama_memory_seq_rm(sys::llama_get_memory(ctx), 0, (n - 1) as i32, -1);
            let r3 = logits_after(ctx, &mut b, toks[n - 1], (n - 1) as i32);
            let d = r1.iter().zip(&r3).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            println!("rollback after prefill chunk: accepted={ok}, max logit diff vs reference {d}");
            return;
        }
        // ---- baseline
        let ctx = make_target(0);
        let t0 = Instant::now();
        let mut next = prefill(ctx, None);
        let pp = t0.elapsed();
        let mut out_base = vec![next];
        let mut margins = vec![f32::INFINITY];
        let mut b = Batch::new(1, 0);
        let t1 = Instant::now();
        let mut pos = toks.len() as i32;
        while out_base.len() < want {
            b.clear();
            b.add(next, pos, true, None);
            assert_eq!(sys::llama_decode(ctx, b.raw), 0);
            pos += 1;
            next = argmax(ctx, 0, n_vocab).0;
            out_base.push(next);
            let row = std::slice::from_raw_parts(sys::llama_get_logits_ith(ctx, 0), n_vocab);
            let top = row[next as usize];
            let second = row.iter().enumerate().filter(|(j, _)| *j as i32 != next).map(|(_, v)| *v).fold(f32::MIN, f32::max);
            margins.push(top - second);
        }
        let base_s = t1.elapsed().as_secs_f64();
        for cut in [16384, 32768, 65536, 100000, 151643] {
            let under = out_base.iter().filter(|t| (**t as usize) < cut).count();
            print!("under {cut}: {:.1}%  ", 100.0 * under as f64 / out_base.len() as f64);
        }
        println!();
        println!("baseline: prefill {:.0} ms, decode {:.1} tok/s", pp.as_secs_f64() * 1e3, (want - 1) as f64 / base_s);
        sys::llama_free(ctx);

        // ---- mtp
        let ctx = make_target(k as u32);
        set_embeddings_nextn(ctx, true, false);
        let mut cp = sys::llama_context_default_params();
        cp.ctx_type = sys::LLAMA_CONTEXT_TYPE_MTP;
        cp.n_ctx = 16384;
        cp.n_batch = 2048;
        cp.n_ubatch = std::env::var("DUB").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
        cp.n_seq_max = 1;
        cp.n_outputs_max = 8;
        cp.type_k = sys::GGML_TYPE_Q8_0;
        cp.type_v = sys::GGML_TYPE_Q8_0;
        cp.flash_attn_type = sys::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.ctx_other = ctx;
        let dctx = sys::llama_init_from_model(model, cp);
        assert!(!dctx.is_null());
        set_embeddings_nextn(dctx, true, true);
        let bsamp = std::env::var("BSAMP").is_ok();
        if bsamp {
            let chain = sys::llama_sampler_chain_init(sys::llama_sampler_chain_default_params());
            sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_greedy());
            assert!(sys::llama_set_sampler(dctx, 0, chain), "backend sampler refused");
        }
        let (mut t_dec, mut t_pick) = (0f64, 0f64);

        let mut pending_h = vec![0f32; n_embd];
        let t0 = Instant::now();
        let mut next = prefill(ctx, Some((dctx, &mut pending_h)));
        let pp = t0.elapsed();
        let mut out = vec![next];
        let mut pos = toks.len() as i32; // position of `next`, not yet decoded by target
        let mut vb = Batch::new(16, 0);
        let mut db = Batch::new(16, n_embd);
        // Catch-up rows owed to the drafter: (token, pos, h).
        let mut owed: Vec<(i32, i32, Vec<f32>)> = Vec::new();
        let (mut proposed, mut accepted_total, mut rounds) = (0usize, 0usize, 0usize);
        let (mut t_draft, mut t_verify) = (0f64, 0f64);
        let t1 = Instant::now();
        while out.len() < want {
            rounds += 1;
            // Draft: catch-up rows + (next, pending_h) at pos, logits.
            let td = Instant::now();
            let mut draft = Vec::new();
            db.clear();
            for (t, p, h) in owed.drain(..) {
                db.add(t, p, false, Some(&h));
            }
            db.add(next, pos, true, Some(&pending_h));
            for step in 0..k {
                let tq = Instant::now();
                assert_eq!(sys::llama_decode(dctx, db.raw), 0);
                sys::llama_synchronize(dctx);
                t_dec += tq.elapsed().as_secs_f64();
                let tq = Instant::now();
                let last = db.raw.n_tokens - 1;
                let (d, p) = if bsamp { (sys::llama_get_sampled_token_ith(dctx, last), 1.0) } else { argmax(dctx, last, n_vocab) };
                t_pick += tq.elapsed().as_secs_f64();
                if p < p_min && step > 0 {
                    break;
                }
                draft.push(d);
                if step + 1 == k {
                    break;
                }
                let h = hrow(dctx, last, n_embd);
                db.clear();
                db.add(d, pos + 1 + step as i32, true, Some(&h));
            }
            if let Ok(mode) = std::env::var("ORACLE") {
                let at = out.len();
                draft = out_base[at.min(want)..(at + k).min(want)].to_vec();
                if mode == "bad" && !draft.is_empty() {
                    let last = draft.len() - 1;
                    draft[last] = (draft[last] + 1) % n_vocab as i32;
                }
            }
            // The drafter keeps position `pos` (a true catch-up row) and loses the guesses.
            sys::llama_memory_seq_rm(sys::llama_get_memory(dctx), 0, pos + 1, -1);
            t_draft += td.elapsed().as_secs_f64();

            // Verify.
            let tv = Instant::now();
            vb.clear();
            vb.add(next, pos, true, None);
            for (i, d) in draft.iter().enumerate() {
                vb.add(*d, pos + 1 + i as i32, true, None);
            }
            assert_eq!(sys::llama_decode(ctx, vb.raw), 0);
            let mut a = 0;
            let mut chosen = argmax(ctx, 0, n_vocab).0;
            while a < draft.len() && draft[a] == chosen {
                out.push(chosen);
                a += 1;
                chosen = argmax(ctx, a as i32, n_vocab).0;
            }
            out.push(chosen);
            proposed += draft.len();
            accepted_total += a;
            // Drafter owes catch-up for accepted drafts at pos+1..=pos+a, with target h shifted by one.
            for i in 0..a {
                owed.push((draft[i], pos + 1 + i as i32, hrow(ctx, i as i32, n_embd)));
            }
            pending_h = hrow(ctx, a as i32, n_embd);
            if a < draft.len() {
                let ok = sys::llama_memory_seq_rm(sys::llama_get_memory(ctx), 0, pos + 1 + a as i32, -1);
                assert!(ok, "ring rollback refused");
            }
            pos += 1 + a as i32;
            next = chosen;
            t_verify += tv.elapsed().as_secs_f64();
        }
        let s = t1.elapsed().as_secs_f64();
        out.truncate(want);
        println!(
            "mtp:      prefill {:.0} ms, decode {:.1} tok/s  ({} rounds, {:.2} tok/round, accepted {}/{} = {:.0}%, draft {:.1} ms/round, verify {:.1} ms/round)",
            pp.as_secs_f64() * 1e3,
            (want - 1) as f64 / s,
            rounds,
            (want - 1) as f64 / rounds as f64,
            accepted_total,
            proposed,
            100.0 * accepted_total as f64 / proposed.max(1) as f64,
            t_draft * 1e3 / rounds as f64,
            t_verify * 1e3 / rounds as f64
        );
        println!("          draft decode {:.2} ms/round, draft pick {:.2} ms/round", t_dec * 1e3 / rounds as f64, t_pick * 1e3 / rounds as f64);
        let same = out_base.iter().zip(&out).position(|(a, b)| a != b);
        match same {
            None => println!("output: IDENTICAL over {want} tokens"),
            Some(i) => println!("output: diverges at token {i}; baseline's top-2 logit gap there {:.3} (median gap {:.2})", margins[i], { let mut m = margins.clone(); m.sort_by(|a, b| a.total_cmp(b)); m[m.len() / 2] }),
        }
        let piece = |ts: &[i32]| {
            let mut s = Vec::new();
            for t in ts {
                let mut buf = [0u8; 64];
                let n = sys::llama_token_to_piece(vocab, *t, buf.as_mut_ptr() as *mut _, 64, 0, true);
                if n > 0 {
                    s.extend_from_slice(&buf[..n as usize]);
                }
            }
            String::from_utf8_lossy(&s).to_string()
        };
        if let Some(i) = same {
            let from = i.saturating_sub(8);
            println!("  base: {:?}", piece(&out_base[from..(i + 8).min(want)]));
            println!("  mtp:  {:?}", piece(&out[from..(i + 8).min(want)]));
        }
        if std::env::var("SHOW").is_ok() {
            println!("---\n{}", piece(&out));
        }
    }
}
