# The Training Loop

Now that we have covered all the core components of MLRust—tensors, autograd, models, layers, loss functions, and optimizers—we can put them all together to train a neural network. The training loop is the process of repeatedly showing the model data, calculating the loss, and updating the model to improve its performance.

## The Steps of a Training Loop

A typical training loop consists of the following steps, repeated for a number of epochs:

1.  **Forward Pass**: Pass the input data through the model to get predictions.
2.  **Compute Loss**: Compare the model's predictions with the actual target values using a loss function.
3.  **Backward Pass**: Call `.backward()` on the loss to compute the gradients of the model's parameters.
4.  **Update Parameters**: Use an optimizer to update the model's parameters based on their gradients.
5.  **Zero Gradients**: Clear the gradients of the parameters to prepare for the next iteration.

## A Complete Example

Here is a complete example of a training loop for a simple linear regression model.

```rust
use mlrust::prelude::*;

fn main() {
    // 1. Create a model
    let mut model = Input::new(1) >> Dense::new(1, 1);

    // 2. Create an optimizer
    let optimizer = SGD::new(0.01);

    // 3. Prepare the data
    let x_train = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4, 1], false);
    let y_train = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[4, 1], false); // y = 2x

    // 4. The training loop
    for epoch in 0..100 {
        // Get the model's parameters
        let mut params = model.params();

        // Forward pass
        let predictions = model.forward(&x_train);

        // Compute loss
        let mut loss = mse_loss(&predictions, &y_train);

        // Backward pass
        loss.backward();

        // Update parameters
        optimizer.step(&mut params);

        // Zero gradients
        optimizer.zero_grad(&mut params);

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss.item());
        }
    }
}
```

### Breaking Down the Example

- **Model**: We create a simple linear model with one input feature and one output feature.
- **Optimizer**: We use `SGD` with a learning rate of `0.01`.
- **Data**: We create some simple data where the target `y` is `2 * x`.
- **Training Loop**: We loop for 100 epochs.
    - Inside the loop, we get the model's parameters, perform the forward pass, compute the loss, and then perform the backward pass.
    - The optimizer then updates the parameters, and we zero out the gradients for the next iteration.

## Training Utilities

MLRust also provides some simple utilities in the `train` module to help with logging and progress tracking during training. These include:

- `ProgressBar`: A simple progress bar.
- `TrainingLogger`: For clean, epoch-based logging.
- `CompactProgress`: A one-line progress indicator.

These are optional but can make the training process more informative and visually appealing.

## Conclusion

This concludes the core documentation for MLRust. You now have a solid understanding of the fundamental concepts and how to use them to build and train neural networks. From here, you can explore the examples in the `examples` directory of the project to see more advanced use cases.