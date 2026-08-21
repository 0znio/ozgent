//! Prints the devices llama.cpp found. Backs `ozgent doctor`.
fn main() {
    for d in ozgent_llama::backend::devices() {
        println!(
            "[{}] {:<10} {:<40} {:.2}/{:.2} GiB free  gpu={}",
            d.index, d.backend, d.description, d.free_gib(), d.total_gib(), d.is_gpu()
        );
    }
    println!("gpu offload supported: {}", ozgent_llama::backend::supports_gpu_offload());
}
