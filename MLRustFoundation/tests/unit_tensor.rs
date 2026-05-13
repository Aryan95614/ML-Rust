use mlrust_foundation::tensor::data::{Tensor, Shape};
use mlrust_foundation::ops::{add, mul, matmul};
use mlrust_foundation::ops::activations::{relu, sigmoid, softmax};
use ndarray::{ArrayD, IxDyn};

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
fn test_shape_from_slice() {
    let shape: Shape = (&[2, 3, 4][..]).into();
    assert_eq!(shape.len(), 3);
}

#[test]
fn test_shape_size() {
    let shape = Shape(vec![2, 3, 4]);
    assert_eq!(shape.size(), 24);
}

#[test]
fn test_gradient_tracking() {
    let a = Tensor::ones(&[2, 2][..], true);
    assert!(a.requires_grad);
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
    let a = Tensor::ones(&[2, 3][..], false);
    let data_b = ArrayD::from_elem(IxDyn(&[3, 2]), 2.0);
    let b = Tensor::new(data_b, &[3, 2][..], false);
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
    assert_eq!(output_data[[3]], 1.0);
}

#[test]
fn test_sigmoid_activation() {
    let data = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let tensor = Tensor::new(data, &[1][..], false);
    let output = sigmoid(&tensor);
    let output_data = output.get_data();
    assert!((output_data[[0]] - 0.5).abs() < 1e-6);
}

#[test]
fn test_softmax_activation() {
    let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();
    let tensor = Tensor::new(data, &[2, 3][..], false);
    let output = softmax(&tensor);
    let output_data = output.get_data();
    assert!(output_data.iter().all(|&x| x > 0.0 && x <= 1.0));
}
