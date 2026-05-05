//! Element-wise tensor multiplication operation

use crate::tensor::data::Tensor;
use crate::tensor::autograd::GradFn;

/// Performs element-wise multiplication of two tensors (Hadamard product)
///
/// # Arguments
///
/// * `a` - First input tensor
/// * `b` - Second input tensor
///
/// # Returns
///
/// A new tensor containing the element-wise product of `a` and `b`
///
/// # Automatic Differentiation
///
/// Implements the chain rule: d(x*y)/dx = y, d(x*y)/dy = x
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let output_data = a.get_data() * b.get_data();
    let requires_grad = a.requires_grad || b.requires_grad;

    let mut output = Tensor::new(output_data, a.shape.clone(), requires_grad);

    if requires_grad {
        output.grad_fn = Some(Box::new(GradFn::Mul {
            input1: a.clone(),
            input2: b.clone(),
            output: output.clone(), // Store output for backward pass
        }));
    }

    output
}

