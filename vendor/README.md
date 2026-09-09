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

## Before publishing

The crates.io packages these were taken from do not carry their upstream
licence files, so the copies here do not either. MIT requires the copyright and
permission notice to travel with the code, so add them:

    curl -o llama-cpp-sys-2/llama.cpp/LICENSE \
      https://raw.githubusercontent.com/ggml-org/llama.cpp/master/LICENSE
    curl -o llama-cpp-2/LICENSE \
      https://raw.githubusercontent.com/utilityai/llama-cpp-rs/main/LICENSE
    cp llama-cpp-2/LICENSE llama-cpp-sys-2/LICENSE

Take them from the upstream repositories rather than writing them out here: the
copyright line is part of the notice, and guessing it is worse than not having
one.
