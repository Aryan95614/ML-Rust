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
            GradFn::Conv2D { input, kernel, stride, padding } => {
                let input_data = input.get_data();
                let kernel_data = kernel.get_data();
                let (n, c_in, h_in, w_in) = (
                    input_data.shape()[0],
                    input_data.shape()[1],
                    input_data.shape()[2],
                    input_data.shape()[3],
                );
                let (c_out, _, k_h, k_w) = (
                    kernel_data.shape()[0],
                    kernel_data.shape()[1],
                    kernel_data.shape()[2],
                    kernel_data.shape()[3],
                );

                let output_h = (h_in + 2 * padding.0 - k_h) / stride.0 + 1;
                let output_w = (w_in + 2 * padding.1 - k_w) / stride.1 + 1;

                // Reshape grad_output from (N, C_out, O_h, O_w) to (N * O_h * O_w, C_out)
                let grad_output_4d = grad_output.clone().into_dimensionality::<ndarray::Ix4>()
                    .map_err(|e| MLError::DimensionError {
                        expected: 4,
                        got: grad_output.ndim(),
                        operation: "Conv2D backward".to_string(),
                    })?;
                let grad_output_permuted = grad_output_4d.permuted_axes([0, 2, 3, 1]); // N, O_h, O_w, C_out
                let grad_output_reshaped = grad_output_permuted
                    .into_shape((n * output_h * output_w, c_out))
                    .map_err(|e| MLError::InvalidOperation(format!("Conv2D backward reshape failed: {}", e)))?
                    .to_owned();

                // Gradient with respect to kernel
                if kernel.requires_grad {
                    let cols = crate::ops::conv2d::im2col(&input_data, k_h, k_w, stride.0, stride.1, padding.0, padding.1);
                    // (C_in * K_h * K_w, N * O_h * O_w) @ (N * O_h * O_w, C_out)
                    // Result: (C_in * K_h * K_w, C_out)
                    let cols_ndim = cols.ndim();
                    let cols_2d = cols.into_dimensionality::<ndarray::Ix2>()
                        .map_err(|_| MLError::DimensionError {
                            expected: 2,
                            got: cols_ndim,
                            operation: "Conv2D kernel gradient".to_string(),
                        })?;
                    let grad_out_ndim = grad_output_reshaped.ndim();
                    let grad_out_2d = grad_output_reshaped.clone().into_dimensionality::<ndarray::Ix2>()
                        .map_err(|_| MLError::DimensionError {
                            expected: 2,
                            got: grad_out_ndim,
                            operation: "Conv2D kernel gradient".to_string(),
                        })?;
                    let grad_kernel_reshaped = cols_2d.t().dot(&grad_out_2d);
                    let grad_kernel_4d = grad_kernel_reshaped
                        .into_shape((c_in, k_h, k_w, c_out))
                        .map_err(|e| MLError::InvalidOperation(format!("Conv2D kernel reshape failed: {}", e)))?;
                    let grad_kernel = grad_kernel_4d
                        .permuted_axes([3, 0, 1, 2]) // Permute to (C_out, C_in, K_h, K_w)
                        .to_owned()
                        .into_dyn();

                    let mut kernel_grad = kernel.grad.lock()?;
                    *kernel_grad = Some(kernel_grad.as_ref().map_or_else(|| grad_kernel.clone(), |g| g + &grad_kernel));
                }

                // Gradient with respect to input
                if input.requires_grad {
                    // Reshape kernel for backprop: (C_out, C_in * K_h * K_w)
                    let kernel_reshaped = kernel_data
                        .into_shape((c_out, c_in * k_h * k_w))
                        .map_err(|e| MLError::InvalidOperation(format!("Conv2D input gradient kernel reshape failed: {}", e)))?;

                    // (N * O_h * O_w, C_out) @ (C_out, C_in * K_h * K_w)
                    // Result: (N * O_h * O_w, C_in * K_h * K_w)
                    let grad_out_reshaped_ndim = grad_output_reshaped.ndim();
                    let grad_out_reshaped_2d = grad_output_reshaped.into_dimensionality::<ndarray::Ix2>()
                        .map_err(|_| MLError::DimensionError {
                            expected: 2,
                            got: grad_out_reshaped_ndim,
                            operation: "Conv2D input gradient".to_string(),
                        })?;
                    let kernel_reshaped_ndim = kernel_reshaped.ndim();
                    let kernel_reshaped_2d = kernel_reshaped.into_dimensionality::<ndarray::Ix2>()
                        .map_err(|_| MLError::DimensionError {
                            expected: 2,
                            got: kernel_reshaped_ndim,
                            operation: "Conv2D input gradient".to_string(),
                        })?;
                    let grad_cols = grad_out_reshaped_2d.dot(&kernel_reshaped_2d).into_dyn();

                    let grad_input = crate::ops::conv2d::col2im(
                        &grad_cols,
                        &input.shape.0, // Original input shape
                        k_h, k_w, stride.0, stride.1, padding.0, padding.1,
                    );

                    let mut input_grad = input.grad.lock()?;
                    *input_grad = Some(input_grad.as_ref().map_or_else(|| grad_input.clone(), |g| g + &grad_input));
                }
            },
            GradFn::MaxPool2D { input, input_shape, max_indices } => {
                if input.requires_grad {
                    let (n, c, h_in, w_in) = (
                        input_shape.0[0],
                        input_shape.0[1],
                        input_shape.0[2],
                        input_shape.0[3],
                    );
                    let (output_h, output_w) = (max_indices.shape()[2], max_indices.shape()[3]); // Output shape of pooling

                    let mut grad_input = ArrayD::zeros(input_shape.to_ixdyn());

                    let grad_output_data = grad_output; // This is the gradient from the next layer

                    let kernel_h = h_in / output_h; // Assuming no padding and stride = kernel_size
                    let kernel_w = w_in / output_w;

                    for n_idx in 0..n {
                        for c_idx in 0..c {
                            for h_out_idx in 0..output_h {
                                for w_out_idx in 0..output_w {
                                    let h_start = h_out_idx * kernel_h;
                                    let w_start = w_out_idx * kernel_w;

                                    let flat_idx = max_indices[[n_idx, c_idx, h_out_idx, w_out_idx]];
                                    let (h_in_idx, w_in_idx) = (
                                        h_start + flat_idx / kernel_w,
                                        w_start + flat_idx % kernel_w,
                                    );

                                    grad_input[[n_idx, c_idx, h_in_idx, w_in_idx]] += grad_output_data[[n_idx, c_idx, h_out_idx, w_out_idx]];
                                }
                            }
                        }
                    }

                    let mut input_grad = input.grad.lock()?;
                    *input_grad = Some(input_grad.as_ref().map_or_else(|| grad_input.clone(), |g| g + &grad_input));
                }
            },
            GradFn::Flatten { input, input_shape } => {
                if input.requires_grad {
                    let grad_input = grad_output.into_shape(input_shape.to_ixdyn())
                        .map_err(|e| MLError::InvalidOperation(format!("Flatten backward reshape failed: {}", e)))?
                        .to_owned();
                    let mut input_grad = input.grad.lock()?;
                    *input_grad = Some(input_grad.as_ref().map_or_else(|| grad_input.clone(), |g| g + &grad_input));
                }
            },
            GradFn::CrossEntropyLoss { logits, targets } => {
                let batch_size = logits.shape.0[0];
                let num_classes = logits.shape.0[1];

                let softmax_output = crate::ops::activations::softmax(logits);
                let softmax_data = softmax_output.get_data();
                let targets_data = targets.get_data();

                let mut grad_logits = ArrayD::zeros(logits.shape.to_ixdyn());

                for i in 0..batch_size {
                    let target_class = targets_data[[i]].round() as usize;
                    for j in 0..num_classes {
                        if j == target_class {
                            grad_logits[[i, j]] = softmax_data[[i, j]] - 1.0;
                        } else {
                            grad_logits[[i, j]] = softmax_data[[i, j]];
                        }
                    }
                }
                let grad_logits = grad_logits / batch_size as f32;

                if logits.requires_grad {
                    let mut logits_grad = logits.grad.lock()?;
                    *logits_grad = Some(logits_grad.as_ref().map_or_else(|| grad_logits.clone(), |g| g + &grad_logits));
                }
            },
            GradFn::None => {},
        }
        Ok(())
    }
}

/// Main backward function for computing gradients
///
/// Performs reverse-mode automatic differentiation starting from the given tensor.
/// Gradients are accumulated in each tensor's grad field.
///
/// # Arguments
///
/// * `tensor` - The output tensor to compute gradients from
///
/// # Errors
///
/// Returns an error if gradient computation fails
pub fn backward(tensor: &Tensor) -> Result<()> {
    let mut topo_order = Vec::new();
    let mut visited = std::collections::HashSet::new();

    fn build_topo(t: &Tensor, topo_order: &mut Vec<Tensor>, visited: &mut std::collections::HashSet<*const Tensor>) {
        let ptr = t as *const Tensor;
        if !visited.insert(ptr) {
            return;
        }

        if let Some(grad_fn) = &t.grad_fn {
            match grad_fn.as_ref() {
                GradFn::Add { input1, input2 } => {
                    build_topo(input1, topo_order, visited);
                    build_topo(input2, topo_order, visited);
                },
                GradFn::Mul { input1, input2, .. } => {
                    build_topo(input1, topo_order, visited);
                    build_topo(input2, topo_order, visited);
                },
                GradFn::MatMul { input1, input2 } => {
                    build_topo(input1, topo_order, visited);
                    build_topo(input2, topo_order, visited);
                },
                GradFn::ReLU { input, .. } => {
                    build_topo(input, topo_order, visited);
                },
                GradFn::Softmax { .. } => {
                    // Softmax typically doesn't have inputs that require grad directly from its GradFn
                    // Its gradient is usually handled by the loss function.
                },
                GradFn::Conv2D { input, kernel, .. } => {
                    build_topo(input, topo_order, visited);
                    build_topo(kernel, topo_order, visited);
                },
                GradFn::MaxPool2D { input, .. } => {
                    build_topo(input, topo_order, visited);
                },
                GradFn::CrossEntropyLoss { logits, targets } => {
                    build_topo(logits, topo_order, visited);
                    // targets don't require grad
                },
                GradFn::Flatten { input, input_shape } => {
                    build_topo(input, topo_order, visited);
                },
                GradFn::Sigmoid { .. } => {
                    // Sigmoid gradient is handled in apply
                },
                GradFn::None => {},
            }
        }
        topo_order.push(t.clone());
    }

    build_topo(tensor, &mut topo_order, &mut visited);

    // Initialize gradient for the output tensor
    let grad_output = ArrayD::ones(tensor.shape.to_ixdyn());
    tensor.set_grad(grad_output.clone())?;

    // Traverse in reverse topological order
    for t in topo_order.iter().rev() {
        if let Some(grad_fn) = &t.grad_fn {
            if let Some(current_grad) = t.get_grad()? {
                grad_fn.apply(current_grad)?; // Pass the accumulated gradient to the GradFn
            }
        }
    }

    Ok(())
}