# The Tensor

The `Tensor` is the central data structure in MLRust. It is a multi-dimensional array that serves as the foundation for all operations in the library. If you are familiar with libraries like PyTorch or TensorFlow, the concept of a tensor will be very familiar.

## What is a Tensor?

A tensor is a generalization of vectors and matrices to an arbitrary number of dimensions. In MLRust, our `Tensor` is specifically a 2D array (a matrix) that wraps an `ndarray::Array2<f32>` and adds support for automatic differentiation.

Each `Tensor` has the following key properties:

- **Data**: The underlying `ndarray::Array2<f32>` that holds the numerical values.
- **Shape**: A tuple `(rows, columns)` that defines the dimensions of the tensor.
- **`requires_grad`**: A boolean flag that indicates whether the tensor should be tracked for gradient computation. If `true`, any operation involving this tensor will be part of the computation graph.
- **`grad`**: An optional `Array2<f32>` that stores the gradient of the tensor after backpropagation.
- **`grad_fn`**: An optional function that points to the operation that created this tensor. This is what allows the computation graph to be traversed backward during backpropagation.

## Creating a Tensor

There are several ways to create a `Tensor` in MLRust.

### From a Vector

You can create a tensor from a flat `Vec<f32>` and a desired shape.

```rust
use mlrust::tensor::Tensor;

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let tensor = Tensor::from_vec(data, &[2, 3], true);

// The tensor will have a shape of (2, 3)
// [[1.0, 2.0, 3.0],
//  [4.0, 5.0, 6.0]]
```

### With Predefined Values

You can also create tensors filled with zeros or ones.

```rust
use mlrust::tensor::Tensor;

// A 2x3 tensor of all zeros
let zeros = Tensor::zeros((2, 3), false);

// A 3x2 tensor of all ones
let ones = Tensor::ones((3, 2), true);
```

### With Random Values

For initializing model parameters, it is common to create tensors with random values drawn from a standard normal distribution.

```rust
use mlrust::tensor::Tensor;

// A 2x2 tensor with random values
let random = Tensor::randn((2, 2), true);
```

## Tensor Operations

The `Tensor` struct supports a variety of operations, all of which are tracked for automatic differentiation.

### Element-wise Operations

These operations are applied to each element of the tensor.

- **Addition**: `tensor.add(&other)`
- **Multiplication**: `tensor.mul(&other)`

### Matrix Operations

- **Matrix Multiplication**: `tensor.matmul(&other)`

### Reduction Operations

These operations reduce a tensor to a scalar value.

- **Sum**: `tensor.sum()`
- **Mean**: `tensor.mean()`

### Activation Functions

- **ReLU**: `tensor.relu()`
- **Sigmoid**: `tensor.sigmoid()`
- **Softmax**: `tensor.softmax()`

Here is an example of how these operations can be chained together:

```rust
use mlrust::tensor::Tensor;

let a = Tensor::from_vec(vec![1.0, 2.0], &[1, 2], true);
let b = Tensor::from_vec(vec![3.0, 4.0], &[2, 1], true);

// c = (a @ b) + 1.0
let c = a.matmul(&b); // (1x2) @ (2x1) -> (1x1)

// The result of the matmul is [[11.0]]
assert_eq!(c.item(), 11.0);
```

Next, we will explore how these operations are tracked for gradient computation in the [Automatic Differentiation](autograd.md) section.