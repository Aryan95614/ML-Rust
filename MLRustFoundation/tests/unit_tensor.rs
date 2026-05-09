use mlrust_foundation::tensor::data::{Tensor, Shape};
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
    assert_eq!(shape.0[0], 2);
    assert_eq!(shape.0[1], 3);
    assert_eq!(shape.0[2], 4);
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
