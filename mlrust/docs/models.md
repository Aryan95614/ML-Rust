# Building Models

In MLRust, a model is a sequence of layers that transforms an input tensor into an output tensor. The library provides two primary ways to build models: using the `Sequential` container directly or using the more intuitive `>>` operator.

## The `Sequential` Container

The `Sequential` struct is a container that holds a vector of layers. It chains them together in the order they are added. The output of one layer becomes the input to the next.

Here is how you can build a simple model using `Sequential` directly:

```rust
use mlrust::layers::{Sequential, Dense, ReLU, Sigmoid};
use mlrust::tensor::Tensor;

// Create a new Sequential model that expects an input of size 784
let mut model = Sequential::new(784);

// Add layers to the model
model.add_layer(Box::new(Dense::new(784, 128)));
model.add_layer(Box::new(ReLU));
model.add_layer(Box::new(Dense::new(128, 10)));
model.add_layer(Box::new(Sigmoid));

// Now you can perform a forward pass
let input = Tensor::randn((32, 784), false); // A batch of 32 samples
let output = model.forward(&input);

assert_eq!(output.shape(), (32, 10));
```

While this approach is explicit and clear, it can be a bit verbose.

## The `>>` Operator: A More Elegant Syntax

To make model building more intuitive and readable, MLRust provides a unique `>>` operator syntax. This operator allows you to chain layers together in a way that visually represents the flow of data through the model.

The `>>` operator is an implementation of Rust's `Shr` (bit shift) trait. It is overloaded for different layer combinations to create a clean, functional pipeline.

Here is the same model built using the `>>` operator:

```rust
use mlrust::layers::{Input, Dense, ReLU, Sigmoid};

let model = Input::new(784)
    >> Dense::new(784, 128)
    >> ReLU
    >> Dense::new(128, 10)
    >> Sigmoid;
```

This syntax is not only more concise but also more closely resembles the structure of the neural network itself. The `Input::new(784)` part serves as a marker to define the input size of the model.

## How the `>>` Operator Works

The `>>` operator is implemented through a series of `Shr` trait implementations:

1.  `Input >> Layer`: The first `>>` operation takes an `Input` and a `Layer` (like `Dense`) and creates a `Sequential` model, adding the layer to it.
2.  `Sequential >> Layer`: Subsequent `>>` operations take the `Sequential` model and another `Layer`, adding the new layer to the sequence.

This allows you to build the entire model as a single expression.

## Getting Model Parameters

Once you have a model, you can get a list of all its trainable parameters (weights and biases) by calling the `.params()` method. This is essential for training, as it provides the optimizer with the tensors it needs to update.

```rust
# use mlrust::layers::{Input, Dense, ReLU, Sigmoid};
# let mut model = Input::new(784)
#     >> Dense::new(784, 128)
#     >> ReLU
#     >> Dense::new(128, 10)
#     >> Sigmoid;
// Get all trainable parameters from the model
let mut params = model.params();

// The optimizer will use this list to update the model
// optimizer.step(&mut params);
```

Now that you know how to build models, let's take a closer look at the different types of [layers](layers.md) available in MLRust.