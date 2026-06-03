# mlrust-core

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Core tensor and autograd primitives shared across the MLRust ecosystem.

## Overview

`mlrust-core` holds the minimal foundation that the other MLRust libraries build on:

- Tensor type with reference-counted data sharing
- Automatic differentiation via a computational graph
- Core operations: add, mul, matmul, relu, sigmoid
- Traits for layers, optimizers, and loss functions

## Design

- Simplicity first. Only the building blocks, nothing domain-specific.
- Shared foundation. Both `mlrust` and `MLRustFoundation` can depend on this.
- Single-threaded by design. The parent libraries are single-threaded.

## Installation

```toml
[dependencies]
mlrust-core = { path = "../mlrust-core" }
```

## Quick example

```rust
use mlrust_core::Tensor;

let a = Tensor::ones((2, 2), true);
let b = Tensor::ones((2, 2), true);

let c = a.add(&b);
let d = c.mul(&a);
let mut loss = d.sum();

loss.backward();

println!("gradient of a: {:?}", a.grad());
```

## Core API

### Tensor creation
- `Tensor::zeros(shape, requires_grad)`
- `Tensor::ones(shape, requires_grad)`
- `Tensor::randn(shape, requires_grad)`
- `Tensor::from_vec(data, shape, requires_grad)`

### Operations (with autograd)
- `tensor.add(other)` - Element-wise addition
- `tensor.mul(other)` - Element-wise multiplication
- `tensor.matmul(other)` - Matrix multiplication
- `tensor.sum()` - Sum all elements
- `tensor.mean()` - Mean of all elements
- `tensor.relu()` - ReLU activation
- `tensor.sigmoid()` - Sigmoid activation
- `tensor.transpose()` - Matrix transpose

### Gradient
- `tensor.backward()` - Kick off backprop (scalar only)
- `tensor.grad()` - Access computed gradient
- `tensor.zero_grad()` - Clear gradient
- `tensor.requires_grad()` - Gradient-tracking flag

## Traits

### Layer
```rust
pub trait Layer {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn params(&mut self) -> Vec<&mut Tensor>;
}
```

### Optimizer
```rust
pub trait Optimizer {
    fn step(&self, params: &mut [&mut Tensor]);
    fn zero_grad(&self, params: &mut [&mut Tensor]);
    fn lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);
}
```

### Loss
```rust
pub trait Loss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Tensor;
}
```

## Layout

```
mlrust-core
├── tensor.rs       # Tensor type with Rc<RefCell<>> for sharing
├── autograd.rs     # GradFn trait and gradient management
├── ops.rs          # Operations with backward implementations
└── traits.rs       # Layer, Optimizer, Loss traits
```

## What's not in core

No specific layer implementations, no optimizer implementations, no loss
function implementations, no training loops, no data loaders. Those belong
in the higher-level libraries (`mlrust` and `MLRustFoundation`).

## Notes on choices

### Rc<RefCell<>>?
Lets tensors be cloned without copying data and supports gradient accumulation
through shared references. Single-threaded, which is fine here.

### 2D only?
Keeps the implementation small. Covers most of what I need. Parent libraries
can reshape for higher dimensions or ship their own ND ops.

### f32 only?
Enough precision for what I'm doing, matches hardware.

## Tests

```bash
cargo test
```

## License

MIT - see LICENSE.

## Used by

- [`mlrust`](../mlrust) - Transformer-focused library
- [`MLRustFoundation`](../MLRustFoundation) - CNN-focused library

