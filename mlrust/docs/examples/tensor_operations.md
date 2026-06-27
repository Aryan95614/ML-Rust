# Example: Tensor Operations

This example demonstrates some of the basic tensor operations available in MLRust. We will create a few tensors and perform some arithmetic operations on them.

```rust
use mlrust::tensor::Tensor;

fn main() {
    // Create two tensors that require gradient computation
    let a = Tensor::from_vec(vec![1.0, 2.0], &[1, 2], true);
    let b = Tensor::from_vec(vec![3.0, 4.0], &[1, 2], true);

    // Perform element-wise addition
    let c = a.add(&b);
    println!("a + b = {:?}", c.data());

    // Perform element-wise multiplication
    let d = a.mul(&b);
    println!("a * b = {:?}", d.data());

    // Create two matrices for matrix multiplication
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let y = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], true);

    // Perform matrix multiplication
    let z = x.matmul(&y);
    println!("x @ y = {:?}", z.data());

    // Compute the sum of a tensor
    let sum_z = z.sum();
    println!("sum(z) = {:?}", sum_z.item());

    // Compute the mean of a tensor
    let mean_z = z.mean();
    println!("mean(z) = {:?}", mean_z.item());
}
```

### Expected Output

```
a + b = [[4.0, 6.0]]
a * b = [[3.0, 8.0]]
x @ y = [[19.0, 22.0], [43.0, 50.0]]
sum(z) = 134.0
mean(z) = 33.5
```

This example shows how the `Tensor` struct is used to perform basic arithmetic and reduction operations. Each of these operations is tracked by the autograd engine, which we will see in the next example.