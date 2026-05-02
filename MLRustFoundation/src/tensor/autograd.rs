//! Automatic differentiation engine
//!
//! This module implements reverse-mode automatic differentiation (backpropagation)
//! for computing gradients of neural network operations.

use std::sync::{Arc, Mutex};
use ndarray::ArrayD;
use crate::tensor::data::{Tensor, Shape};
use crate::error::{MLError, Result};

/// Gradient function representing an operation in the computational graph
///
/// Each variant stores the inputs needed to compute gradients during
/// the backward pass.
#[derive(Clone, Debug)]
pub enum GradFn {
    Add {
        input1: Tensor,
        input2: Tensor,
    },
    Mul {
        input1: Tensor,
        input2: Tensor,
        output: Tensor,
    },
    MatMul {
        input1: Tensor,
        input2: Tensor,
    },
    ReLU {
        input: Tensor,
        output: Tensor, // Store output to compute gradient
    },
    Softmax {
        output: Tensor, // Store output to compute gradient
    },
    Sigmoid {
        output: Tensor, // Store output to compute gradient
    },
    Conv2D {
        input: Tensor,
        kernel: Tensor,
        stride: crate::ops::conv2d::Stride,
        padding: crate::ops::conv2d::Padding,
    },
    MaxPool2D {
        input: Tensor,
        input_shape: Shape, // Original input shape
        max_indices: ArrayD<usize>, // Indices of max values
    },
    CrossEntropyLoss {
        logits: Tensor,
        targets: Tensor,
    },
    Flatten {
        input: Tensor,
        input_shape: Shape,
    },
    // Placeholder for other operations
    None,
}

impl GradFn {
    /// Applies the gradient function to compute input gradients
    ///
    /// # Arguments
    ///
    /// * `grad_output` - The gradient flowing from the next layer
    ///
    /// # Errors
    ///
    /// Returns an error if lock acquisition fails or dimension conversion fails
    pub fn apply(&self, grad_output: ArrayD<f32>) -> Result<()> {
        match self {
            GradFn::Add { input1, input2 } => {
                // Gradients for addition are simply passed through
                if input1.requires_grad {
                    let mut input1_grad = input1.grad.lock()?;
                    *input1_grad = Some(input1_grad.as_ref().map_or_else(|| grad_output.clone(), |g| g + &grad_output));
                }
                if input2.requires_grad {
                    let mut input2_grad = input2.grad.lock()?;
                    *input2_grad = Some(input2_grad.as_ref().map_or_else(|| grad_output.clone(), |g| g + &grad_output));
                }
            },
            GradFn::Mul { input1, input2, output } => {
                // d(x*y)/dx = y, d(x*y)/dy = x
                let output_data = output.get_data();
                let input1_data = input1.get_data();
                let input2_data = input2.get_data();

                if input1.requires_grad {
                    let grad_input1 = &grad_output * &input2_data;
                    let mut input1_grad = input1.grad.lock()?;
                    *input1_grad = Some(input1_grad.as_ref().map_or_else(|| grad_input1.clone(), |g| g + &grad_input1));
                }
                if input2.requires_grad {
                    let grad_input2 = &grad_output * &input1_data;
                    let mut input2_grad = input2.grad.lock()?;
                    *input2_grad = Some(input2_grad.as_ref().map_or_else(|| grad_input2.clone(), |g| g + &grad_input2));
                }
            },
            GradFn::MatMul { input1, input2 } => {
                // d(A @ B)/dA = grad_output @ B.T
                // d(A @ B)/dB = A.T @ grad_output
                let input1_data = input1.get_data();
                let input2_data = input2.get_data();

                if input1.requires_grad {
                    let grad_out_ndim = grad_output.ndim();
                    let grad_out_2d = grad_output.clone().into_dimensionality::<ndarray::Ix2>()
                        .map_err(|_| MLError::DimensionError {
                            expected: 2,
                            got: grad_out_ndim,
                            operation: "MatMul backward".to_string(),
                        })?;
                    let input2_ndim = input2_data.ndim();
                    let input2_2d = input2_data.clone().into_dimensionality::<ndarray::Ix2>()
                        .map_err(|_| MLError::DimensionError {
                            expected: 2,
                            got: input2_ndim,
                            operation: "MatMul backward".to_string(),
                        })?;
                    let grad_input1 = grad_out_2d.dot(&input2_2d.t()).into_dyn();
                    let mut input1_grad = input1.grad.lock()?;
                    *input1_grad = Some(input1_grad.as_ref().map_or_else(|| grad_input1.clone(), |g| g + &grad_input1));
                }
                if input2.requires_grad {
                    let input1_ndim = input1_data.ndim();
                    let input1_2d = input1_data.into_dimensionality::<ndarray::Ix2>()
                        .map_err(|_| MLError::DimensionError {
                            expected: 2,
                            got: input1_ndim,
                            operation: "MatMul backward".to_string(),
                        })?;
                    let grad_out_ndim = grad_output.ndim();
                    let grad_out_2d = grad_output.into_dimensionality::<ndarray::Ix2>()
                        .map_err(|_| MLError::DimensionError {
                            expected: 2,
                            got: grad_out_ndim,
                            operation: "MatMul backward".to_string(),
                        })?;
                    let grad_input2 = input1_2d.t().dot(&grad_out_2d).into_dyn();
                    let mut input2_grad = input2.grad.lock()?;
                    *input2_grad = Some(input2_grad.as_ref().map_or_else(|| grad_input2.clone(), |g| g + &grad_input2));
                }
            },
            GradFn::ReLU { input, output } => {
                // d(ReLU(x))/dx = 1 if x > 0, else 0
                let input_data = input.get_data();
                let grad_input = grad_output * input_data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                if input.requires_grad {
                    let mut input_grad = input.grad.lock()?;
                    *input_grad = Some(input_grad.as_ref().map_or_else(|| grad_input.clone(), |g| g + &grad_input));
                }
            },
            GradFn::Softmax { output } => {
                // Softmax gradient: ∂L/∂x_i = softmax_i * (∂L/∂y_i - Σ_j(∂L/∂y_j * softmax_j))
                // This implements the full Jacobian-vector product for standalone softmax
                let output_data = output.get_data();

                // Compute: sum over last axis of (grad_output * softmax_output)
                let grad_times_output = &grad_output * &output_data;

                // Sum along the last axis (class dimension)
                let sum_axis = output_data.ndim() - 1;
                let sum_grad_output = grad_times_output.sum_axis(ndarray::Axis(sum_axis));

                // Broadcast the sum back to the original shape
                let mut sum_broadcasted = sum_grad_output.clone();
                for _ in 0..grad_output.shape()[sum_axis] - 1 {
                    sum_broadcasted = ndarray::concatenate(
                        ndarray::Axis(sum_axis),
                        &[sum_broadcasted.view(), sum_grad_output.view()]
                    ).unwrap();
                }

                // grad_input = softmax * (grad_output - sum_broadcasted)
                let grad_input = &output_data * &(&grad_output - &sum_broadcasted.into_dyn());

                if output.requires_grad {
                    let mut output_grad = output.grad.lock()?;
                    *output_grad = Some(output_grad.as_ref().map_or_else(
                        || grad_input.clone(),
                        |g| g + &grad_input
                    ));
                }
            },
            GradFn::Sigmoid { output } => {
                // Gradient of sigmoid: output * (1 - output) * grad_output
                let output_data = output.get_data();
                let grad = output_data.mapv(|x| x * (1.0 - x));

                if output.requires_grad {
                    let grad_input = &grad * &grad_output;
                    let mut output_grad = output.grad.lock()?;
                    *output_grad = Some(output_grad.as_ref().map_or_else(
                        || grad_input.clone(),
                        |g| g + &grad_input
                    ));
                }
            },
