//! Element-wise tensor addition operation

use crate::tensor::data::Tensor;
use crate::tensor::autograd::GradFn;

/// Performs element-wise addition of two tensors
///
/// # Arguments
///
/// * `a` - First input tensor
/// * `b` - Second input tensor
///
/// # Returns
///
/// A new tensor containing the element-wise sum of `a` and `b`
///
/// # Automatic Differentiation
///
/// If either input requires gradients, the output will track the computation
/// for backpropagation. The gradient is distributed equally to both inputs.
///
/// # Examples
///
/// ```no_run
/// use mlrust::ops::add;
/// use mlrust::tensor::Tensor;
/// use ndarray::ArrayD;
///
/// let a = Tensor::ones(&[2, 2], true);
/// let b = Tensor::ones(&[2, 2], true);
/// let c = add(&a, &b);  // c = [[2, 2], [2, 2]]
/// ```
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let output_data = a.get_data() + b.get_data();
    let requires_grad = a.requires_grad || b.requires_grad;

    let mut output = Tensor::new(output_data, a.shape.clone(), requires_grad);

    if requires_grad {
        output.grad_fn = Some(Box::new(GradFn::Add {
            input1: a.clone(),
            input2: b.clone(),
        }));
    }

    output
}

