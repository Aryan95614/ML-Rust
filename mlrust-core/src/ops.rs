//! Core tensor operations with automatic differentiation

use ndarray::{Array2, Axis};
use std::sync::Arc;

use crate::{GradFn, Tensor};

/// Helper to reduce gradients for broadcasting
fn reduce_grad(grad: &Array2<f32>, target_shape: (usize, usize)) -> Array2<f32> {
    let mut result = grad.clone();
    let grad_shape = grad.dim();

    if grad_shape != target_shape {
        if target_shape.0 == 1 && grad_shape.0 > 1 {
            result = result.sum_axis(Axis(0)).insert_axis(Axis(0));
        }
        if target_shape.1 == 1 && grad_shape.1 > 1 {
            result = result.sum_axis(Axis(1)).insert_axis(Axis(1));
        }
    }
    result
}

// ============================================================================
// Addition
// ============================================================================

struct AddBackward {
    lhs: Tensor,
    rhs: Tensor,
}

impl GradFn for AddBackward {
    fn backward(&self, grad: &Array2<f32>) -> Vec<Array2<f32>> {
        let grad_lhs = reduce_grad(grad, self.lhs.shape());
        let grad_rhs = reduce_grad(grad, self.rhs.shape());

        if self.lhs.requires_grad() {
            self.lhs.accumulate_grad(grad_lhs.clone());
            if let Some(grad_fn) = self.lhs.grad_fn() {
                grad_fn.backward(&grad_lhs);
            }
        }

        if self.rhs.requires_grad() {
            self.rhs.accumulate_grad(grad_rhs.clone());
            if let Some(grad_fn) = self.rhs.grad_fn() {
                grad_fn.backward(&grad_rhs);
            }
        }

        vec![grad_lhs, grad_rhs]
    }
}

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let result_data = &lhs.data() + &rhs.data();
    let requires_grad = lhs.requires_grad() || rhs.requires_grad();
    let result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        result.set_grad_fn(Arc::new(AddBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        }));
    }

    result
}

// ============================================================================
// Multiplication (element-wise)
// ============================================================================

struct MulBackward {
    lhs: Tensor,
    rhs: Tensor,
    lhs_data: Array2<f32>,
    rhs_data: Array2<f32>,
}

impl GradFn for MulBackward {
    fn backward(&self, grad: &Array2<f32>) -> Vec<Array2<f32>> {
        let grad_lhs = grad * &self.rhs_data;
        let grad_rhs = grad * &self.lhs_data;

        let grad_lhs = reduce_grad(&grad_lhs, self.lhs.shape());
        let grad_rhs = reduce_grad(&grad_rhs, self.rhs.shape());

        if self.lhs.requires_grad() {
            self.lhs.accumulate_grad(grad_lhs.clone());
            if let Some(grad_fn) = self.lhs.grad_fn() {
                grad_fn.backward(&grad_lhs);
            }
        }

        if self.rhs.requires_grad() {
            self.rhs.accumulate_grad(grad_rhs.clone());
            if let Some(grad_fn) = self.rhs.grad_fn() {
                grad_fn.backward(&grad_rhs);
            }
        }

        vec![grad_lhs, grad_rhs]
    }
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let result_data = &lhs.data() * &rhs.data();
    let requires_grad = lhs.requires_grad() || rhs.requires_grad();
    let result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        result.set_grad_fn(Arc::new(MulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            lhs_data: lhs.data(),
            rhs_data: rhs.data(),
        }));
    }

    result
}

// ============================================================================
// Matrix Multiplication
// ============================================================================

struct MatMulBackward {
    lhs: Tensor,
    rhs: Tensor,
}

impl GradFn for MatMulBackward {
    fn backward(&self, grad: &Array2<f32>) -> Vec<Array2<f32>> {
        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        let grad_lhs = grad.dot(&rhs_data.t());
        let grad_rhs = lhs_data.t().dot(grad);

        if self.lhs.requires_grad() {
            self.lhs.accumulate_grad(grad_lhs.clone());
            if let Some(grad_fn) = self.lhs.grad_fn() {
                grad_fn.backward(&grad_lhs);
            }
        }

        if self.rhs.requires_grad() {
            self.rhs.accumulate_grad(grad_rhs.clone());
            if let Some(grad_fn) = self.rhs.grad_fn() {
                grad_fn.backward(&grad_rhs);
            }
        }

        vec![grad_lhs, grad_rhs]
    }
}

pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let result_data = lhs.data().dot(&rhs.data());
    let requires_grad = lhs.requires_grad() || rhs.requires_grad();
    let result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        result.set_grad_fn(Arc::new(MatMulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        }));
    }

    result
}

// ============================================================================
// Reduction operations
// ============================================================================

struct SumBackward {
    input: Tensor,
}

impl GradFn for SumBackward {
    fn backward(&self, grad: &Array2<f32>) -> Vec<Array2<f32>> {
        let input_shape = self.input.shape();
        let grad_input = Array2::from_elem(input_shape, grad[[0, 0]]);

        if self.input.requires_grad() {
            self.input.accumulate_grad(grad_input.clone());
            if let Some(grad_fn) = self.input.grad_fn() {
                grad_fn.backward(&grad_input);
            }
        }

        vec![grad_input]
    }
}

pub fn sum(input: &Tensor) -> Tensor {
    let sum_value = input.data().sum();
    let result_data = Array2::from_elem((1, 1), sum_value);
    let requires_grad = input.requires_grad();
    let result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        result.set_grad_fn(Arc::new(SumBackward {
            input: input.clone(),
        }));
    }

    result
}

struct MeanBackward {
    input: Tensor,
}

impl GradFn for MeanBackward {
    fn backward(&self, grad: &Array2<f32>) -> Vec<Array2<f32>> {
        let input_shape = self.input.shape();
        let n = (input_shape.0 * input_shape.1) as f32;
        let grad_input = Array2::from_elem(input_shape, grad[[0, 0]] / n);

        if self.input.requires_grad() {
            self.input.accumulate_grad(grad_input.clone());
            if let Some(grad_fn) = self.input.grad_fn() {
                grad_fn.backward(&grad_input);
            }
        }

        vec![grad_input]
    }
}

pub fn mean(input: &Tensor) -> Tensor {
    let mean_value = input.data().mean().unwrap_or(0.0);
    let result_data = Array2::from_elem((1, 1), mean_value);
    let requires_grad = input.requires_grad();
    let result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        result.set_grad_fn(Arc::new(MeanBackward {
            input: input.clone(),
        }));
    }

    result
}

// ============================================================================
// Activations
// ============================================================================

struct ReLUBackward {
    input: Tensor,
}

impl GradFn for ReLUBackward {
    fn backward(&self, grad: &Array2<f32>) -> Vec<Array2<f32>> {
        let input_data = self.input.data();
