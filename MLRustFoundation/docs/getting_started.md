# Getting Started with MLRustFoundation

A quick 5-minute guide to building your first neural network with MLRustFoundation.

## Installation

```bash
cargo new my_ml_project
cd my_ml_project
```

Add to your `Cargo.toml`:
```toml
[dependencies]
mlrust_foundation = { path = "../MLRustFoundation" }
ndarray = "0.15"
rand = "0.8"
```

## Your First Neural Network

Create a simple XOR network:

```rust
use mlrust_foundation::layers::*;
use mlrust_foundation::tensor::Tensor;
use mlrust_foundation::ops::add;
use ndarray::{ArrayD, IxDyn};
use rand::thread_rng;

fn main() {
    // Create a simple 2-layer network
    let mut rng = thread_rng();

    let model = Input::image(1, 1, 2)  // 2 inputs
        >> Flatten::new()
        >> Dense::new(2, 4, &mut rng).expect("Failed to create layer")
        >> ReLU::new()
        >> Dense::new(4, 1, &mut rng).expect("Failed to create layer")
        >> Sigmoid::new();

    // Create input
    let data = ArrayD::from_shape_vec(
        IxDyn(&[1, 1, 1, 2]),
        vec![1.0, 0.0]
    ).unwrap();
    let input = Tensor::new(data, &[1, 1, 1, 2][..], false);

    // Forward pass
    let output = model.forward(&input);
    println!("Output: {:?}", output.get_data());
}
```

## Training Your First Model

See the MNIST examples for complete training loops:

```bash
# Train on MNIST digits
cargo run --example mnist_dense

# Train a CNN
cargo run --example mnist_cnn
```

## Next Steps

1. Read the [API Documentation](../README.md#-api-overview)
2. Explore the [examples directory](../examples/)
3. Check out the [architecture guide](../README.md#-architecture)

## Common Patterns

### Creating Tensors
```rust
// From zeros/ones
let zeros = Tensor::zeros(&[2, 3][..], false);
let ones = Tensor::ones(&[2, 3][..], true);  // with gradient tracking

// From data
let data = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
let tensor = Tensor::new(data, &[2, 2][..], true);
```

### Building Models
```rust
// Use the >> operator to chain layers
let model = Input::image(1, 28, 28)
    >> Conv2D::new(1, 16, (3, 3), Stride(1, 1), Padding(1, 1), &mut rng).unwrap()
    >> ReLU::new()
    >> MaxPool2D::new((2, 2))
    >> Flatten::new()
    >> Dense::new(16 * 14 * 14, 10, &mut rng).unwrap()
    >> Softmax::new();
```

### Training Loop
```rust
use mlrust_foundation::optim::sgd::SGD;
use mlrust_foundation::loss::cross_entropy_loss;

let mut optimizer = SGD::new(0.01, Some(0.9));

for epoch in 0..10 {
    let output = model.forward(&input);
    let loss = cross_entropy_loss(&output, &labels);

    optimizer.zero_grad(model.params());
    loss.backward().expect("Backward pass failed");
    optimizer.step(model.params());
}
```

## Troubleshooting

**Problem**: "Shape mismatch" error
**Solution**: Check that your layer dimensions match. The output of one layer must match the input of the next.

**Problem**: "Lock error"
**Solution**: Make sure you're not trying to access the same tensor from multiple threads.

**Problem**: Tests failing
**Solution**: Run `cargo clean && cargo build` to rebuild from scratch.

## Resources

- [Full API Docs](https://docs.rs/mlrust_foundation) - Run `cargo doc --open`
- [Examples](../examples/) - Complete working examples
- [Tests](../tests/) - See unit tests for usage patterns

