# MLRustFoundation

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A small ML library I'm writing in Rust while learning the language.
Tensors with reverse-mode autograd, a few layers, SGD, and MNIST examples.

## Features

- Tensor with reverse-mode autodiff (computational graph tracking)
- Layers: Dense, Conv2D, MaxPool2D, Flatten, ReLU, Sigmoid, Softmax
- Optimizer: SGD with optional momentum
- Cross-entropy loss
- Error types with `Result` returns where it matters
- MNIST data loader (downloads + parses IDX gzip)
- A CLI binary and a Criterion bench

## Installation

```toml
[dependencies]
mlrust_foundation = { path = "." }
```

## Quick Start

### A simple dense network for MNIST

```rust
use mlrust_foundation::layers::*;
use mlrust_foundation::tensor::Tensor;
use mlrust_foundation::loss::cross_entropy_loss;
use rand::thread_rng;

let mut rng = thread_rng();
let model = Input::image(1, 28, 28)
    >> Flatten::new()
    >> Dense::new(784, 128, &mut rng).unwrap()
    >> ReLU::new()
    >> Dense::new(128, 10, &mut rng).unwrap()
    >> Softmax::new();

let output = model.forward(&input);
let loss = cross_entropy_loss(&output, &targets);
loss.backward().expect("Backpropagation failed");
```

### A small CNN

```rust
use mlrust_foundation::layers::*;
use mlrust_foundation::ops::conv2d::{Stride, Padding};

let mut rng = thread_rng();
let model = Input::image(1, 28, 28)
    >> Conv2D::new(1, 16, (3, 3), Stride(1, 1), Padding(1, 1), &mut rng).unwrap()
    >> ReLU::new()
    >> MaxPool2D::new((2, 2))
    >> Conv2D::new(16, 32, (3, 3), Stride(1, 1), Padding(1, 1), &mut rng).unwrap()
    >> ReLU::new()
    >> MaxPool2D::new((2, 2))
    >> Flatten::new()
    >> Dense::new(32 * 7 * 7, 128, &mut rng).unwrap()
    >> ReLU::new()
    >> Dense::new(128, 10, &mut rng).unwrap()
    >> Softmax::new();
```

## Examples

```bash
# Train a dense net on MNIST
cargo run --example mnist_dense

# Train a CNN on MNIST
cargo run --example mnist_cnn

# Or use the CLI
cargo run --bin mlrust -- train --model dense --epochs 5
```

## Architecture

- `tensor` - `Tensor` type with `Arc<Mutex<>>` data and optional gradient
- `ops` - add, mul, matmul, activations, conv2d (with im2col)
- `layers` - layer trait plus the implementations above, composed with `>>`
- `loss` - cross-entropy loss
- `optim` - SGD (with momentum)
- `train` - a small `Trainer` that ties everything together
- `data` - MNIST loader
- `bin/mlrust` - CLI

## Autograd

Reverse-mode. Each op produces a `GradFn` node referring to its inputs. Calling
`.backward()` walks the graph in reverse topological order and accumulates
gradients on every tensor where `requires_grad` is true.

```rust
let a = Tensor::ones(&[2, 2][..], true);
let b = Tensor::ones(&[2, 2][..], true);
let c = add(&a, &b);
let loss = c.sum();
loss.backward()?;
let grad_a = a.get_grad()?;
```

## Tests

```bash
cargo test
```

`tests/unit_tensor.rs` covers tensor creation, basic ops, activations,
shape conversions, and a couple of backward-pass sanity checks.

## Performance

Pure Rust, CPU only. `Arc<Mutex<>>` for shared tensor data. `f32` throughout.

## Known limitations

- MatMul only for 2D matrices
- No broadcasting in the ops (I just use explicit reshapes)
- No GPU
- `f32` only

## License

MIT - see [LICENSE](LICENSE).
