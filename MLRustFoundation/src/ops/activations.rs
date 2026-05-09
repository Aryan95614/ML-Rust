//! Activation functions for neural networks
//!
//! Provides common non-linear activation functions including:
//! - ReLU: Rectified Linear Unit
//! - Softmax: Normalized exponential function
//! - Sigmoid: Logistic function

use ndarray::ArrayD;
use crate::tensor::data::Tensor;
use crate::tensor::autograd::GradFn;

/// Applies the Rectified Linear Unit (ReLU) activation function
///
/// ReLU(x) = max(0, x)
///
/// # Arguments
///
/// * `input` - Input tensor
///
/// # Returns
///
/// A new tensor with ReLU applied element-wise
///
/// # Automatic Differentiation
///
/// Gradient: ∂ReLU(x)/∂x = 1 if x > 0, else 0
pub fn relu(input: &Tensor) -> Tensor {
    let output_data = input.get_data().mapv(|x| x.max(0.0));
    let requires_grad = input.requires_grad;

    let mut output = Tensor::new(output_data, input.shape.clone(), requires_grad);

    if requires_grad {
        output.grad_fn = Some(Box::new(GradFn::ReLU {
            input: input.clone(),
            output: output.clone(), // Store output for backward pass
        }));
    }

    output
}

/// Applies the Softmax activation function along the last axis
///
/// Softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
///
/// # Arguments
///
/// * `input` - Input tensor
///
/// # Returns
///
/// A new tensor with Softmax applied along the last axis.
/// Output values are normalized probabilities that sum to 1.
///
/// # Automatic Differentiation
///
/// Implements the full Jacobian-vector product:
/// ∂L/∂x_i = softmax_i * (∂L/∂y_i - Σ_j(∂L/∂y_j * softmax_j))
///
/// # Note
///
/// Uses numerically stable computation by subtracting the maximum value
/// before exponentiation to prevent overflow.
pub fn softmax(input: &Tensor) -> Tensor {
    let input_data = input.get_data();
    let mut output_data = ArrayD::zeros(input_data.shape());

    let axis = input_data.ndim() - 1; // Apply softmax along the last axis

    input_data.axis_iter(ndarray::Axis(axis)).zip(output_data.axis_iter_mut(ndarray::Axis(axis)))
        .for_each(|(input_slice, mut output_slice)| {
            let max_val = input_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_values: Vec<f32> = input_slice.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_values.iter().sum();

            for (i, &exp_val) in exp_values.iter().enumerate() {
                output_slice[i] = exp_val / sum_exp;
            }
        });

    let requires_grad = input.requires_grad;
    let mut output = Tensor::new(output_data, input.shape.clone(), requires_grad);

    if requires_grad {
        output.grad_fn = Some(Box::new(GradFn::Softmax {
            output: output.clone(), // Store output for backward pass
        }));
    }

    output
}

/// Applies the Sigmoid (logistic) activation function
///
/// Sigmoid(x) = 1 / (1 + exp(-x))
///
/// # Arguments
///
/// * `input` - Input tensor
///
/// # Returns
///
/// A new tensor with Sigmoid applied element-wise.
/// Output values are in the range (0, 1).
///
/// # Automatic Differentiation
///
/// Gradient: ∂Sigmoid(x)/∂x = sigmoid(x) * (1 - sigmoid(x))
///
/// # Use Cases
///
/// Commonly used for binary classification and as a gating mechanism
/// in recurrent neural networks.
pub fn sigmoid(input: &Tensor) -> Tensor {
    let output_data = input.get_data().mapv(|x| 1.0 / (1.0 + (-x).exp()));
    let requires_grad = input.requires_grad;

    let mut output = Tensor::new(output_data, input.shape.clone(), requires_grad);

    if requires_grad {
        output.grad_fn = Some(Box::new(GradFn::Sigmoid {
            output: output.clone(), // Store output for backward pass
        }));
    }

    output
}

// Note: Inline tests have been moved to tests/unit_tensor.rs
// See tests/unit_tensor.rs for activation tests
