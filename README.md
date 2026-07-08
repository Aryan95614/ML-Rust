# ML-Rust

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A family of machine learning libraries written from scratch in Rust, with zero ML dependencies.

## Libraries

| Library | Focus | Size |
|---------|-------|------|
| [`mlrust-core`](mlrust-core) | Shared tensor & autograd foundation | ~1,200 lines |
| [`mlrust`](mlrust) | Transformers & NLP | ~10,000 lines |
| [`MLRustFoundation`](MLRustFoundation) | CNNs & computer vision | ~2,600 lines |

See [ECOSYSTEM.md](ECOSYSTEM.md) for the full architecture, feature comparison, and integration examples across the three libraries.

## Quick start

```toml
[dependencies]
mlrust-core = { path = "../mlrust-core" }
```

```rust
use mlrust_core::Tensor;

let a = Tensor::ones((2, 2), true);
let b = Tensor::ones((2, 2), true);

let c = a.add(&b);
let mut loss = c.sum();
loss.backward();

println!("gradient of a: {:?}", a.grad());
```

Each crate has its own README with install instructions, quick-start examples,
and a rundown of what is (and isn't) implemented yet:

- [`mlrust-core/README.md`](mlrust-core/README.md)
- [`mlrust/README.md`](mlrust/README.md)
- [`mlrust-transformers/README.md`](mlrust-transformers/README.md)
- [`MLRustFoundation/README.md`](MLRustFoundation/README.md)

## Building

Each library is its own Cargo project:

```bash
cd mlrust-core && cargo test
cd MLRustFoundation && cargo test
cd mlrust-transformers && cargo test
```

## Philosophy

Clarity over performance. These are learning projects — every component is
written to be readable and easy to follow rather than optimized for
production workloads.

## License

MIT - see [LICENSE](LICENSE).
