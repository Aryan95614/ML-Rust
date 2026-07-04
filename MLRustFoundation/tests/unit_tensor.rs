//! Unit tests for tensor operations

use mlrust_foundation::tensor::data::{Tensor, Shape};
use mlrust_foundation::ops::{add, mul, matmul};
use mlrust_foundation::ops::activations::{relu, sigmoid, softmax};
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_tensor_creation() {
    let data = ArrayD::zeros(IxDyn(&[2, 3]));
    let tensor = Tensor::new(data, &[2, 3][..], false);

    assert_eq!(tensor.shape.len(), 2);
    assert_eq!(tensor.shape.0[0], 2);
    assert_eq!(tensor.shape.0[1], 3);
    assert!(!tensor.requires_grad);
}

#[test]
fn test_tensor_zeros() {
    let tensor = Tensor::zeros(&[3, 4][..], false);
    let data = tensor.get_data();

    assert_eq!(data.shape(), &[3, 4]);
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn test_tensor_ones() {
    let tensor = Tensor::ones(&[2, 2][..], false);
    let data = tensor.get_data();

    assert_eq!(data.shape(), &[2, 2]);
    assert!(data.iter().all(|&x| x == 1.0));
}

#[test]
fn test_add_operation() {
    let a = Tensor::ones(&[2, 2][..], false);
    let b = Tensor::ones(&[2, 2][..], false);
    let c = add(&a, &b);

    let data = c.get_data();
    assert!(data.iter().all(|&x| x == 2.0));
}

#[test]
fn test_mul_operation() {
    let data_a = ArrayD::from_elem(IxDyn(&[2, 2]), 3.0);
    let a = Tensor::new(data_a, &[2, 2][..], false);

    let data_b = ArrayD::from_elem(IxDyn(&[2, 2]), 4.0);
    let b = Tensor::new(data_b, &[2, 2][..], false);

    let c = mul(&a, &b);
    let data = c.get_data();

    assert!(data.iter().all(|&x| x == 12.0));
}

#[test]
fn test_matmul_operation() {
    // Create a 2x3 matrix of ones
    let a = Tensor::ones(&[2, 3][..], false);

    // Create a 3x2 matrix of twos
    let data_b = ArrayD::from_elem(IxDyn(&[3, 2]), 2.0);
    let b = Tensor::new(data_b, &[3, 2][..], false);

    // Result should be 2x2 matrix where each element is 3*2 = 6
    let c = matmul(&a, &b);
    let data = c.get_data();

    assert_eq!(data.shape(), &[2, 2]);
    assert!(data.iter().all(|&x| x == 6.0));
}

#[test]
fn test_relu_activation() {
    let data = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-2.0, -1.0, 0.0, 1.0]).unwrap();
    let tensor = Tensor::new(data, &[4][..], false);

    let output = relu(&tensor);
    let output_data = output.get_data();

    assert_eq!(output_data[[0]], 0.0);
    assert_eq!(output_data[[1]], 0.0);
    assert_eq!(output_data[[2]], 0.0);
    assert_eq!(output_data[[3]], 1.0);
}

#[test]
fn test_sigmoid_activation() {
    let data = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let tensor = Tensor::new(data, &[1][..], false);

    let output = sigmoid(&tensor);
    let output_data = output.get_data();

    // sigmoid(0) = 0.5
    assert!((output_data[[0]] - 0.5).abs() < 1e-6);
}

#[test]
fn test_softmax_activation() {
    let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();
    let tensor = Tensor::new(data, &[2, 3][..], false);

    let output = softmax(&tensor);
    let output_data = output.get_data();

    // Check that all values are positive
    assert!(output_data.iter().all(|&x| x > 0.0));

    // Softmax output should be between 0 and 1
    assert!(output_data.iter().all(|&x| x >= 0.0 && x <= 1.0));

    // First row should have increasing values (input 1, 2, 3)
    assert!(output_data[[0, 0]] < output_data[[0, 1]]);
    assert!(output_data[[0, 1]] < output_data[[0, 2]]);

    // Second row should have similar values since input is all 1s
    let val0 = output_data[[1, 0]];
    let val1 = output_data[[1, 1]];
    let val2 = output_data[[1, 2]];
    // Values should be close (within 10%)
    assert!((val0 - val1).abs() / val0 < 0.1);
    assert!((val1 - val2).abs() / val1 < 0.1);
}

#[test]
fn test_gradient_tracking() {
    let a = Tensor::ones(&[2, 2][..], true);
    let b = Tensor::ones(&[2, 2][..], true);
    let c = add(&a, &b);

    assert!(c.requires_grad);
    assert!(c.grad_fn.is_some());
}

#[test]
fn test_no_gradient_tracking() {
    let a = Tensor::ones(&[2, 2][..], false);
    let b = Tensor::ones(&[2, 2][..], false);
    let c = add(&a, &b);

    assert!(!c.requires_grad);
    assert!(c.grad_fn.is_none());
}

#[test]
fn test_backward_simple() {
    let a = Tensor::ones(&[2, 2][..], true);
    let b = Tensor::ones(&[2, 2][..], true);
    let c = add(&a, &b);

    let result = c.backward();
    assert!(result.is_ok());

    // Check that gradients were computed
    let grad_a = a.get_grad();
    assert!(grad_a.is_ok());
    assert!(grad_a.unwrap().is_some());
}

#[test]
fn test_backward_without_requires_grad() {
    let a = Tensor::ones(&[2, 2][..], false);

    let result = a.backward();
    assert!(result.is_err());
}

#[test]
fn test_shape_from_slice() {
    let shape: Shape = (&[2, 3, 4][..]).into();
    assert_eq!(shape.len(), 3);
    assert_eq!(shape.0[0], 2);
    assert_eq!(shape.0[1], 3);
    assert_eq!(shape.0[2], 4);
}

#[test]
fn test_shape_from_tuple() {
    let shape: Shape = (10, 20).into();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape.0[0], 10);
    assert_eq!(shape.0[1], 20);
}

#[test]
fn test_shape_size() {
    let shape = Shape(vec![2, 3, 4]);
    assert_eq!(shape.size(), 24);
}
