//! Does the embedder rank the right memory first, and how fast is it where?
//!
//! Eight questions, each worded differently from the one note that answers it,
//! against twelve notes of the kind a conversation leaves behind. Scored as the
//! rank of the right note (MRR, top-1), with the embedder as ozgent used to
//! run it (`OZGENT_EMBED_PLAIN=1`: no special tokens, no query instruction)
//! and as it runs now — then timed on the CPU and the GPU.
//!
//! cargo run --release -p ozgent-llama -p ozgent-cli --features ozgent-cli/cuda \
//!   --example embedbench -- <embedding-model.gguf>
use ozgent_llama::embed::{Embedder, Role};
use std::time::Instant;

const NOTES: [&str; 20] = [
    "We moved the database backups to run every night at 2am and they now go to the NAS in the hallway.",
    "My daughter's school play is on the 14th and I promised to help paint the stage set.",
    "The API was timing out under load; we added a connection pool and response times dropped from 900ms to 120ms.",
    "I'm allergic to penicillin, so the doctor prescribed azithromycin instead.",
    "The landlord agreed to fix the leaking bathroom tap before the end of the month.",
    "For the quarterly report, revenue from the Berlin office grew 18% while Madrid stayed flat.",
    "I switched my commute to cycling three days a week and lost four kilos since March.",
    "The Rust build kept recompiling llama.cpp because the feature flags differed between two cargo invocations.",
    "Grandma's recipe uses browned butter and a pinch of cardamom in the apple pie crust.",
    "Our flight to Lisbon leaves at 06:40 from terminal 2, and the hotel check-in is after 3pm.",
    "The cat has been refusing her dry food, the vet suggested switching to a wet diet.",
    "I set the thermostat schedule to drop to 17 degrees at night to cut the heating bill.",
    // Near misses: each shares a topic with one of the answers above.
    "We looked at backing up to an S3 bucket but the storage costs were too high.",
    "The dog is due for his rabies vaccination in May.",
    "My flight to Berlin last month was delayed by two hours at the gate.",
    "cargo test takes about three minutes on the CI runners.",
    "My son's football match is on Saturday morning at the park.",
    "I'm allergic to cats, which makes visiting my aunt difficult.",
    "We made the landing page load faster by compressing the hero images.",
    "Mum's bread recipe uses rye flour and a long overnight proof.",
];

/// (question, index of the note that answers it)
const QUESTIONS: [(&str, usize); 8] = [
    ("Where do the nightly copies of our data end up?", 0),
    ("Which antibiotic can I not take?", 3),
    ("How did we make the service faster when it was struggling with traffic?", 2),
    ("When is the kid's performance at school?", 1),
    ("What secret ingredient goes into the family dessert?", 8),
    ("Why did compiling take forever every time?", 7),
    ("What did the pet doctor recommend for feeding?", 10),
    ("What time do we need to be at the airport for Portugal?", 9),
];

fn score(embedder: &Embedder) -> (f64, usize, f64) {
    let notes: Vec<String> = NOTES.iter().map(|s| s.to_string()).collect();
    let docs = embedder.embed_as(Role::Document, &notes).expect("embedding notes");
    let qs: Vec<String> = QUESTIONS.iter().map(|(q, _)| q.to_string()).collect();
    let queries = embedder.embed_as(Role::Query, &qs).expect("embedding questions");
    let mut mrr = 0.0;
    let mut top1 = 0;
    let mut margin = 0.0f64;
    for ((_, want), q) in QUESTIONS.iter().zip(&queries) {
        let mut ranked: Vec<(usize, f32)> =
            docs.iter().enumerate().map(|(i, d)| (i, q.iter().zip(d).map(|(a, b)| a * b).sum())).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let rank = ranked.iter().position(|(i, _)| i == want).unwrap() + 1;
        mrr += 1.0 / rank as f64;
        top1 += (rank == 1) as usize;
        // How far the right note stands above the best wrong one: rank hides
        // a narrow win, and a narrow win is one paraphrase from a loss.
        let right = ranked.iter().find(|(i, _)| i == want).unwrap().1;
        let wrong = ranked.iter().filter(|(i, _)| i != want).map(|(_, s)| *s).fold(f32::MIN, f32::max);
        margin += (right - wrong) as f64;
    }
    let n = QUESTIONS.len() as f64;
    (mrr / n, top1, margin / n)
}

fn time(embedder: &Embedder, texts: &[String], rounds: usize) -> f64 {
    let _ = embedder.embed_as(Role::Document, texts);
    let t = Instant::now();
    for _ in 0..rounds {
        let _ = embedder.embed_as(Role::Document, texts);
    }
    t.elapsed().as_secs_f64() * 1000.0 / rounds as f64
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: embedbench <model.gguf>");
    let path = std::path::Path::new(&path);

    for plain in [true, false] {
        // SAFETY: single-threaded; read by the next load.
        unsafe { std::env::set_var("OZGENT_EMBED_PLAIN", if plain { "1" } else { "0" }) };
        let e = Embedder::load(path, 99).expect("loading");
        let (mrr, top1, margin) = score(&e);
        println!(
            "{}: MRR {mrr:.3}, right note first {top1}/{}, mean margin over the best wrong note {margin:+.3}",
            if plain { "as before (no EOS, no instruction)" } else { "fixed (EOS + query instruction)" },
            QUESTIONS.len()
        );
    }

    unsafe { std::env::set_var("OZGENT_EMBED_PLAIN", "0") };
    // Split into micro-batches, a long text must give the vector it gives
    // in one: 6k tokens fits the 8k working context whole or not at all.
    {
        let mut mid = String::new();
        for i in 0..280 {
            mid.push_str(&format!("Entry {i}: the quarterly figures were reconciled and filed. "));
        }
        let e = Embedder::load_with(path, 99, 0).expect("loading");
        let a = e.embed_as(Role::Document, &[mid.clone()]).unwrap().remove(0);
        let b = e.embed_as(Role::Document, &[mid, "x".repeat(40000)]).unwrap().remove(0);
        let cos: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        println!("same text, in a working context vs a grown one: cosine {cos:.6}");
    }
    // A long note whose answer is at the end: what a 512-token cut loses.
    let mut long = String::new();
    for i in 0..900 {
        long.push_str(&format!("Day {i} of the renovation: the plasterers finished another wall and the skip was emptied. "));
    }
    long.push_str("Most importantly, the spare house key is hidden under the blue flowerpot by the back door.");
    let docs: Vec<String> = std::iter::once(long).chain(NOTES.iter().map(|s| s.to_string())).collect();
    let q = vec!["Where did we leave the spare key?".to_string()];
    for window in [512u32, 0] {
        let e = Embedder::load_with(path, 99, window).expect("loading");
        let t = Instant::now();
        let d = e.embed_as(Role::Document, &docs).unwrap();
        let qv = e.embed_as(Role::Query, &q).unwrap().remove(0);
        let mut ranked: Vec<(usize, f32)> = d.iter().enumerate().map(|(i, v)| (i, qv.iter().zip(v).map(|(a, b)| a * b).sum())).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let rank = ranked.iter().position(|(i, _)| *i == 0).unwrap() + 1;
        println!(
            "long note, {} window: the key note ranks {rank} of {} ({:.0} ms for all {} notes)",
            if window == 0 { "the model's whole".to_string() } else { format!("{window}-token") },
            docs.len(),
            t.elapsed().as_secs_f64() * 1000.0,
            docs.len()
        );
    }
    let message = vec![QUESTIONS[2].0.to_string()];
    let reply = vec![NOTES.join(" ")];
    let eight: Vec<String> = NOTES[..8].iter().map(|s| s.to_string()).collect();
    for (label, layers) in [("GPU", 99u32), ("CPU", 0)] {
        let e = Embedder::load(path, layers).expect("loading");
        println!(
            "{label}: short message {:6.1} ms   300-token reply {:6.1} ms   8 notes {:6.1} ms",
            time(&e, &message, 10),
            time(&e, &reply, 5),
            time(&e, &eight, 5)
        );
    }
}
