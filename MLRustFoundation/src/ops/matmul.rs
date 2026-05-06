//! Matrix multiplication operation

use crate::tensor::data::{Tensor, Shape};
use crate::tensor::autograd::GradFn;

/// Performs matrix multiplication of two 2D tensors
///
/// # Arguments
///
/// * `a` - First input tensor (shape: M × K)
/// * `b` - Second input tensor (shape: K × N)
///
/// # Returns
///
/// A new tensor containing the matrix product (shape: M × N)
///
/// # Panics
///
/// Panics if:
/// - Either input is not 2D
/// - Inner dimensions don't match (a.shape[1] != b.shape[0])
///
/// # Automatic Differentiation
///
/// Implements gradients:
/// - ∂L/∂A = ∂L/∂output @ B^T
/// - ∂L/∂B = A^T @ ∂L/∂output
///
/// # Note
///
/// Currently only supports 2D matrices. Broadcasting and higher dimensions
/// are planned for future releases.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.get_data();
    let b_data = b.get_data();

    // Assuming 2D for now for simplicity with ndarray::dot
    // TODO: Handle broadcasting and higher dimensions
    assert_eq!(a_data.ndim(), 2, "Matmul input A must be 2D");
    assert_eq!(b_data.ndim(), 2, "Matmul input B must be 2D");
    assert_eq!(a_data.shape()[1], b_data.shape()[0], "Matmul dimensions mismatch");

    // Convert to 2D arrays for dot product
    let a_2d = a_data.into_dimensionality::<ndarray::Ix2>().unwrap();
    let b_2d = b_data.into_dimensionality::<ndarray::Ix2>().unwrap();
    let output_2d = a_2d.dot(&b_2d);
    let output_data = output_2d.into_dyn();
    let output_shape = Shape(vec![output_data.shape()[0], output_data.shape()[1]]);
    let requires_grad = a.requires_grad || b.requires_grad;

    let mut output = Tensor::new(output_data, output_shape, requires_grad);

    if requires_grad {
        output.grad_fn = Some(Box::new(GradFn::MatMul {
            input1: a.clone(),
            input2: b.clone(),
        }));
    }

    output
}