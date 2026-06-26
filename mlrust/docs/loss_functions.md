# Loss Functions

A loss function, also known as a cost function or objective function, is a crucial component in training a neural network. It measures how well the model's predictions match the actual target values. The goal of training is to minimize the value of the loss function.

In MLRust, loss functions are implemented as standalone functions that take the model's predictions and the ground truth targets as input and return a scalar tensor representing the loss.

## Mean Squared Error (MSE)

The Mean Squared Error (`mse_loss`) is a common loss function used for regression tasks, where the goal is to predict a continuous value.

It is calculated as the average of the squared differences between the predicted values and the actual values:

`MSE = (1/n) * Σ(prediction - target)²`

### Usage

```rust
use mlrust::loss::mse_loss;
use mlrust::tensor::Tensor;

let predictions = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], true);
let targets = Tensor::from_vec(vec![1.5, 2.5, 3.5], &[1, 3], false);

// Calculate the MSE loss
let loss = mse_loss(&predictions, &targets);

// The loss will be a scalar tensor
assert_eq!(loss.shape(), (1, 1));
```

## Binary Cross-Entropy (BCE)

The Binary Cross-Entropy (`bce_loss`) is the standard loss function for binary classification tasks, where the goal is to classify an input into one of two categories.

It is designed to be used with models that output a probability between 0 and 1 (typically from a `Sigmoid` activation function).

### Usage

```rust
use mlrust::loss::bce_loss;
use mlrust::tensor::Tensor;

// Predictions from a model with a sigmoid activation
let predictions = Tensor::from_vec(vec![0.9, 0.2, 0.8], &[1, 3], true);

// Target labels (0 or 1)
let targets = Tensor::from_vec(vec![1.0, 0.0, 1.0], &[1, 3], false);

// Calculate the BCE loss
let loss = bce_loss(&predictions, &targets);

assert_eq!(loss.shape(), (1, 1));
```

For numerical stability, the `bce_loss` function in MLRust clamps the predictions to a small epsilon value away from 0 and 1 to avoid taking the logarithm of zero.

With a way to measure the model's performance, the next step is to learn how to update the model to improve it. This is where [optimizers](optimizers.md) come in.