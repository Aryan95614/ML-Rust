# Automatic Differentiation

Automatic differentiation (autograd) is the core of what makes a deep learning library like MLRust work. It is the mechanism that automatically computes the gradients of a set of outputs (like a loss function) with respect to a set of inputs (like the model's parameters). These gradients are then used by optimizers to update the parameters and improve the model.

## The Computation Graph

Every operation performed on tensors in MLRust creates a computation graph. This is a directed acyclic graph (DAG) where the nodes are tensors and the edges are the operations (or functions) that produced them.

When you perform an operation, a new tensor is created, and this new tensor keeps a reference to the function that created it via the `grad_fn` property. This function, in turn, holds references to the input tensors.

For example, consider the expression `c = a.add(b)`:

- `a` and `b` are leaf nodes of the graph (tensors we created).
- A new tensor `c` is created.
- `c` has a `grad_fn` that points to an `AddBackward` function.
- This `AddBackward` function holds references to `a` and `b`.

This creates a chain that can be traversed backward from any tensor to its ancestors.

## Backpropagation

Backpropagation is the process of traversing this computation graph backward to compute gradients. It is initiated by calling the `.backward()` method on a scalar tensor (usually the loss).

Here is what happens when you call `.backward()`:

1.  **Initialize Gradient**: The gradient of the tensor on which `.backward()` is called is initialized to `1.0`.
2.  **Call `grad_fn`**: The `grad_fn` of the current tensor is called. This function computes the gradient of the output with respect to its inputs.
3.  **Propagate Gradients**: The computed gradients are then passed down to the input tensors of the function.
4.  **Accumulate Gradients**: The gradients are accumulated in the `.grad` property of each tensor.
5.  **Recurse**: This process is repeated recursively until all leaf nodes (tensors with `requires_grad=true` that were not created by an operation) are reached.

### An Example

Let's trace the backpropagation for `loss = (x * w).sum()`:

1.  **Forward Pass**:
    - `p = x.mul(w)`: Creates a tensor `p` with a `grad_fn` pointing to `MulBackward`, which holds references to `x` and `w`.
    - `loss = p.sum()`: Creates a tensor `loss` with a `grad_fn` pointing to `SumBackward`, which holds a reference to `p`.

2.  **Backward Pass** (`loss.backward()`):
    - The gradient of `loss` is set to `1.0`.
    - The `grad_fn` of `loss` (`SumBackward`) is called. It computes the gradient with respect to `p` and calls `p.backward()`.
    - The `grad_fn` of `p` (`MulBackward`) is called. It computes the gradients with respect to `x` and `w`.
    - The gradients are accumulated in `x.grad` and `w.grad`.

## The `GradFn` Trait

At the heart of the autograd engine is the `GradFn` trait. Each operation that supports autograd has a corresponding struct that implements this trait (e.g., `AddBackward`, `MulBackward`, `MatMulBackward`).

This trait has a single method:

```rust
// Simplified for explanation
pub trait GradFn {
    fn backward(&self, grad: &Array2<f32>) -> Vec<Array2<f32>>;
}
```

The `backward` method takes the gradient from the output of the operation and computes the gradients for its inputs.

For example, for element-wise multiplication `c = a * b`, the gradients are:

- `grad_a = grad_c * b`
- `grad_b = grad_c * a`

The `MulBackward` struct stores `a` and `b` and uses them to compute these gradients when its `backward` method is called.

By chaining these `GradFn` implementations together, MLRust can automatically compute the gradients for any complex sequence of operations.

Next, let's see how to use these concepts to [build models](models.md).