# Example: Model Construction

This example demonstrates how to construct a neural network model in MLRust using the `Sequential` container and the `>>` operator. We will build a simple multi-layer perceptron (MLP).

## Building a Model with `Sequential`

First, let's build a model by adding layers to a `Sequential` container directly.

```rust
use mlrust::layers::{Sequential, Dense, ReLU, Sigmoid};
use mlrust::tensor::Tensor;

fn main() {
    // Define the input size
    let input_size = 10;

    // Create a Sequential model
    let mut model = Sequential::new(input_size);
    model.add_layer(Box::new(Dense::new(10, 20)));
    model.add_layer(Box::new(ReLU));
    model.add_layer(Box::new(Dense::new(20, 5)));
    model.add_layer(Box::new(Sigmoid));

    // Perform a forward pass
    let input = Tensor::randn((1, input_size), false);
    let output = model.forward(&input);

    println!("Model output shape: {:?}", output.shape());
}
```

## Building a Model with the `>>` Operator

Now, let's build the exact same model using the more intuitive `>>` operator.

```rust
use mlrust::layers::{Input, Dense, ReLU, Sigmoid};
use mlrust::tensor::Tensor;

fn main() {
    // Define the input size
    let input_size = 10;

    // Create the model using the >> operator
    let model = Input::new(input_size)
        >> Dense::new(10, 20)
        >> ReLU
        >> Dense::new(20, 5)
        >> Sigmoid;

    // Perform a forward pass
    let input = Tensor::randn((1, input_size), false);
    let output = model.forward(&input);

    println!("Model output shape: {:?}", output.shape());
}
```

### Expected Output (for both examples)

```
Model output shape: (1, 5)
```

As you can see, both methods produce the same model, but the `>>` operator provides a more concise and readable way to define the architecture of the network. This syntax is designed to mimic the visual flow of data through the layers of the model.