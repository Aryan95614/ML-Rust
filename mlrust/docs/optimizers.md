# Optimizers

An optimizer is an algorithm that adjusts the parameters of a neural network (its weights and biases) to minimize the loss function. After the gradients have been computed during the backpropagation step, the optimizer uses these gradients to update the parameters.

## Stochastic Gradient Descent (SGD)

MLRust provides a basic implementation of one of the most fundamental optimization algorithms: Stochastic Gradient Descent (`SGD`).

SGD updates the parameters in the opposite direction of their gradients, scaled by a learning rate. The update rule for a single parameter `p` is:

`p = p - learning_rate * p.grad`

### Creating an Optimizer

You create an `SGD` optimizer by specifying a learning rate.

```rust
use mlrust::optim::SGD;

// Create an SGD optimizer with a learning rate of 0.01
let optimizer = SGD::new(0.01);
```

The learning rate is a hyperparameter that controls how large of a step the optimizer takes during each update. Choosing a good learning rate is crucial for effective training.

### Using the Optimizer

The optimizer has two main methods that are used in a training loop:

- `step()`: Updates the model's parameters using their gradients.
- `zero_grad()`: Clears the gradients of the parameters.

Here is how they are typically used:

```rust
# use mlrust::layers::{Input, Dense};
# use mlrust::optim::SGD;
# use mlrust::loss::mse_loss;
# use mlrust::tensor::Tensor;
# let mut model = Input::new(1) >> Dense::new(1, 1);
# let optimizer = SGD::new(0.01);
# let x = Tensor::from_vec(vec![1.0], &[1, 1], false);
# let y = Tensor::from_vec(vec![1.0], &[1, 1], false);
// 1. Get the model's parameters
let mut params = model.params();

// 2. Perform a forward pass and compute the loss
let predictions = model.forward(&x);
let mut loss = mse_loss(&predictions, &y);

// 3. Compute gradients
loss.backward();

// 4. Update the parameters
optimizer.step(&mut params);

// 5. Zero out the gradients for the next iteration
optimizer.zero_grad(&mut params);
```

It is important to call `zero_grad()` after each optimization step because, by default, gradients are accumulated in MLRust. If you don't clear them, the gradients from the next backward pass will be added to the existing ones, leading to incorrect updates.

Now that we have all the components, let's put them all together in the [training loop](training.md).