mod autograd;
mod data;
mod ops;

pub use autograd::GradFn;
pub use data::Tensor;

use ndarray::Array2;

/// Type alias for tensor shape (rows, columns)
pub type Shape = (usize, usize);

/// Helper function to validate scalar shape
pub(crate) fn is_scalar(shape: Shape) -> bool {
    shape == (1, 1)
}

/// Helper function to extract value at position from Array2
pub(crate) fn extract_scalar(data: &Array2<f32>, shape: Shape) -> f32 {
    assert!(
        is_scalar(shape),
        "item() can only be called on scalar tensors, got shape {:?}",
        shape
    );
    data[[0, 0]]
}
