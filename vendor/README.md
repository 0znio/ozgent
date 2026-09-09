# Vendored dependencies

Two crates are vendored here rather than taken from crates.io. Both are patched
copies, and the reason for each is in its own `OZGENT.md`.

| directory | upstream | licence |
|---|---|---|
| `llama-cpp-2` | [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs) | MIT OR Apache-2.0 |
| `llama-cpp-sys-2` | [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs) | MIT OR Apache-2.0 |
| `llama-cpp-sys-2/llama.cpp` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT |
| `llama-cpp-sys-2/llama.cpp/vendor/*` | cpp-httplib, nlohmann/json, miniaudio, stb | see each directory |

These are third-party works redistributed here. ozgent's own MIT licence in the
repository root covers ozgent, not them.

## The licence files here

The crates.io packages these were taken from do not carry their upstream
licence files, so these were fetched from the upstream repositories — MIT and
Apache-2.0 both require the notice to travel with the code, and the copyright
line is part of the notice.

| file | from |
|---|---|
| `llama-cpp-sys-2/llama.cpp/LICENSE` | [ggml-org/llama.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/LICENSE) |
| `llama-cpp-{2,sys-2}/LICENSE-MIT` | [utilityai/llama-cpp-rs](https://raw.githubusercontent.com/utilityai/llama-cpp-rs/main/LICENSE-MIT) |
| `llama-cpp-{2,sys-2}/LICENSE-APACHE` | [utilityai/llama-cpp-rs](https://raw.githubusercontent.com/utilityai/llama-cpp-rs/main/LICENSE-APACHE) |

llama-cpp-rs is dual-licensed, so both files are kept: taking only one would be
choosing on the user's behalf, which is the opposite of what "MIT OR Apache-2.0"
means.

The nested `llama.cpp/vendor/*` directories (cpp-httplib, nlohmann/json,
miniaudio, stb) carry their own notices where upstream shipped them.
