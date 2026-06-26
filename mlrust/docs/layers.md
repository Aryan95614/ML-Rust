# Layers

Layers are the building blocks of neural networks. In MLRust, every layer is a struct that implements the `Layer` trait. This trait defines the essential behavior of a layer: a `forward` method for processing input data and a `params` method for providing its trainable parameters.

## The `Layer` Trait

The `Layer` trait is the common interface for all layers in the library. It looks like this:

```rust
pub trait Layer {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn params(&mut self) -> Vec<&mut Tensor>;
}
```

- `forward`: Takes an input tensor and returns an output tensor.
- `params`: Returns a vector of mutable references to the layer's trainable parameters. If a layer has no trainable parameters (like an activation function), it returns an empty vector.

## Dense Layer

The `Dense` layer, also known as a fully connected layer, is one of the most common layers in neural networks. It performs a linear transformation on the input data: `output = input @ weights + bias`.

### Creating a Dense Layer

You create a `Dense` layer by specifying the number of input and output features.

```rust
use mlrust::layers::Dense;

// A dense layer with 784 input features and 128 output features
let dense_layer = Dense::new(784, 128);
```

The weights of the `Dense` layer are initialized using Kaiming (He) initialization, which is a common practice for layers followed by a ReLU activation. The biases are initialized to zeros.

## Activation Layers

Activation layers apply a non-linear function to the output of a previous layer. This non-linearity is what allows neural networks to learn complex patterns. Activation layers do not have any trainable parameters.

### ReLU

The Rectified Linear Unit (`ReLU`) is one of the most popular activation functions. It is defined as `f(x) = max(0, x)`.

```rust
use mlrust::layers::ReLU;

let relu_layer = ReLU;
```

### Sigmoid

The `Sigmoid` activation function squashes its input into a range between 0 and 1. It is often used in the output layer of a binary classification model. It is defined as `f(x) = 1 / (1 + e^(-x))`.

```rust
use mlrust::layers::Sigmoid;

let sigmoid_layer = Sigmoid;
```

### Identity

The `Identity` layer simply returns its input without any changes. It is useful as a placeholder or for linear regression models where no activation is needed on the output.

```rust
use mlrust::layers::Identity;

let identity_layer = Identity;
```

## LayerNorm

`LayerNorm` (Layer Normalization) is a normalization layer that helps stabilize the training of deep neural networks. It normalizes the inputs across the feature dimension.

```rust
use mlrust::layers::LayerNorm;

// A layer normalization layer for an input with 128 features
let layernorm_layer = LayerNorm::new(128);
```

`LayerNorm` has two learnable parameters, `gamma` (scale) and `beta` (shift), which are initialized to ones and zeros, respectively.

Now that you are familiar with the available layers, let's look at the [loss functions](loss_functions.md) used to train models.