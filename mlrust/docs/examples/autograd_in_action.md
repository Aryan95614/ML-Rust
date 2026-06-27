# Example: Autograd in Action

This example provides a step-by-step demonstration of the automatic differentiation engine (autograd) in MLRust. We will define a simple expression, compute its value, and then perform backpropagation to get the gradients.

Consider the expression `d = a * b + c`.

The partial derivatives are:
- `∂d/∂a = b`
- `∂d/∂b = a`
- `∂d/∂c = 1`

Let's verify this with MLRust.

```rust
use mlrust::tensor::Tensor;

fn main() {
    // Create three tensors
    let a = Tensor::from_vec(vec![2.0], &[1, 1], true);
    let b = Tensor::from_vec(vec![3.0], &[1, 1], true);
    let c = Tensor::from_vec(vec![4.0], &[1, 1], true);

    // Compute d = a * b + c
    let ab = a.mul(&b);
    let mut d = ab.add(&c);

    // At this point, d has a value of 2.0 * 3.0 + 4.0 = 10.0
    println!("d = {:?}", d.item());

    // Perform backpropagation
    d.backward();

    // Now we can inspect the gradients of a, b, and c
    println!("∂d/∂a = {:?}", a.grad().unwrap());
    println!("∂d/∂b = {:?}", b.grad().unwrap());
    println!("∂d/∂c = {:?}", c.grad().unwrap());
}
```

### Expected Output

```
d = 10.0
∂d/∂a = [[3.0]]
∂d/∂b = [[2.0]]
∂d/∂c = [[1.0]]
```

### How It Works

1.  **Forward Pass**:
    - `ab = a.mul(&b)`: A `MulBackward` function is created and attached to `ab`. This function stores references to `a` and `b`.
    - `d = ab.add(&c)`: An `AddBackward` function is created and attached to `d`. This function stores references to `ab` and `c`.

2.  **Backward Pass (`d.backward()`)**:
    - The gradient of `d` is initialized to `1.0`.
    - The `grad_fn` of `d` (`AddBackward`) is called. It passes the gradient of `d` (which is `1.0`) to both `ab` and `c`.
        - The gradient of `c` is accumulated, so `c.grad` becomes `1.0`.
    - The `grad_fn` of `ab` (`MulBackward`) is now called with the incoming gradient (`1.0`). It computes the gradients for `a` and `b`:
        - `grad_a = grad_ab * b.data() = 1.0 * 3.0 = 3.0`
        - `grad_b = grad_ab * a.data() = 1.0 * 2.0 = 2.0`
    - The gradients are accumulated in `a.grad` and `b.grad`.

This example illustrates how the computation graph is built during the forward pass and then traversed during the backward pass to compute the gradients of the leaf tensors.