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
        let grad_input = grad * &input_data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

        if self.input.requires_grad() {
            self.input.accumulate_grad(grad_input.clone());
            if let Some(grad_fn) = self.input.grad_fn() {
                grad_fn.backward(&grad_input);
            }
        }

        vec![grad_input]
    }
}

pub fn relu(input: &Tensor) -> Tensor {
    let result_data = input.data().mapv(|x| x.max(0.0));
    let requires_grad = input.requires_grad();
    let result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        result.set_grad_fn(Arc::new(ReLUBackward {
            input: input.clone(),
        }));
    }

    result
}

struct SigmoidBackward {
    input: Tensor,
    output: Array2<f32>,
}

impl GradFn for SigmoidBackward {
    fn backward(&self, grad: &Array2<f32>) -> Vec<Array2<f32>> {
        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        let grad_input = grad * &self.output.mapv(|y| y * (1.0 - y));

        if self.input.requires_grad() {
            self.input.accumulate_grad(grad_input.clone());
            if let Some(grad_fn) = self.input.grad_fn() {
                grad_fn.backward(&grad_input);
            }
        }

        vec![grad_input]
    }
}

pub fn sigmoid(input: &Tensor) -> Tensor {
    let result_data = input.data().mapv(|x| 1.0 / (1.0 + (-x).exp()));
    let requires_grad = input.requires_grad();
    let result = Tensor::new(result_data.clone(), requires_grad);

    if requires_grad {
        result.set_grad_fn(Arc::new(SigmoidBackward {
            input: input.clone(),
            output: result_data,
        }));
    }

    result
}

// ============================================================================
// Log (natural logarithm)
// ============================================================================

struct LogBackward {
    input: Tensor,
}

impl GradFn for LogBackward {
    fn backward(&self, grad: &Array2<f32>) -> Vec<Array2<f32>> {
        // d/dx log(x) = 1/x
        let input_data = self.input.data();
        let grad_input = grad / &input_data;

        if self.input.requires_grad() {
            self.input.accumulate_grad(grad_input.clone());
            if let Some(grad_fn) = self.input.grad_fn() {
                grad_fn.backward(&grad_input);
            }
        }

        vec![grad_input]
    }
}

pub fn log(input: &Tensor) -> Tensor {
    let result_data = input.data().mapv(|x| x.ln());
    let requires_grad = input.requires_grad();
    let result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        result.set_grad_fn(Arc::new(LogBackward {
            input: input.clone(),
        }));
    }

    result
}

// ============================================================================
// Negation
// ============================================================================

struct NegBackward {
    input: Tensor,
}

impl GradFn for NegBackward {
    fn backward(&self, grad: &Array2<f32>) -> Vec<Array2<f32>> {
        // d/dx (-x) = -1
        let grad_input = grad.mapv(|x| -x);

        if self.input.requires_grad() {
            self.input.accumulate_grad(grad_input.clone());
            if let Some(grad_fn) = self.input.grad_fn() {
                grad_fn.backward(&grad_input);
            }
        }

        vec![grad_input]
    }
}

pub fn neg(input: &Tensor) -> Tensor {
    let result_data = input.data().mapv(|x| -x);
    let requires_grad = input.requires_grad();
    let result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        result.set_grad_fn(Arc::new(NegBackward {
            input: input.clone(),
        }));
    }

    result
}

// ============================================================================
// Clamp
// ============================================================================

struct ClampBackward {
    input: Tensor,
    min: f32,
    max: f32,
}

impl GradFn for ClampBackward {
    fn backward(&self, grad: &Array2<f32>) -> Vec<Array2<f32>> {
        // Gradient flows through only where input is in [min, max]
        let input_data = self.input.data();
        let mut grad_input = grad.clone();

        // Zero out gradients where input was clamped
        for (g, x) in grad_input.iter_mut().zip(input_data.iter()) {
            if *x <= self.min || *x >= self.max {
                *g = 0.0;
            }
        }

        if self.input.requires_grad() {
            self.input.accumulate_grad(grad_input.clone());
            if let Some(grad_fn) = self.input.grad_fn() {
                grad_fn.backward(&grad_input);
            }
        }

        vec![grad_input]
    }
}

pub fn clamp(input: &Tensor, min: f32, max: f32) -> Tensor {
    let result_data = input.data().mapv(|x| x.max(min).min(max));
    let requires_grad = input.requires_grad();
    let result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        result.set_grad_fn(Arc::new(ClampBackward {
            input: input.clone(),
            min,
            max,
        }));
    }

    result
}

// ============================================================================
// Transpose
// ============================================================================

struct TransposeBackward {
    input: Tensor,
}

impl GradFn for TransposeBackward {
    fn backward(&self, grad: &Array2<f32>) -> Vec<Array2<f32>> {
        let grad_input = grad.t().to_owned();

        if self.input.requires_grad() {
            self.input.accumulate_grad(grad_input.clone());
            if let Some(grad_fn) = self.input.grad_fn() {
                grad_fn.backward(&grad_input);
            }
        }

        vec![grad_input]
    }
}

pub fn transpose(input: &Tensor) -> Tensor {
    let result_data = input.data().t().to_owned();
    let requires_grad = input.requires_grad();
    let result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        result.set_grad_fn(Arc::new(TransposeBackward {
            input: input.clone(),
        }));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_backward() {
        let a = Tensor::from_vec(vec![2.0], &[1, 1], true);
        let b = Tensor::from_vec(vec![3.0], &[1, 1], true);
        let mut c = add(&a, &b);

        c.backward();

        assert_eq!(a.grad().unwrap()[[0, 0]], 1.0);
        assert_eq!(b.grad().unwrap()[[0, 0]], 1.0);
    }

    #[test]
    fn test_mul_backward() {
        let a = Tensor::from_vec(vec![2.0], &[1, 1], true);
        let b = Tensor::from_vec(vec![3.0], &[1, 1], true);
        let mut c = mul(&a, &b);

        c.backward();

        assert_eq!(a.grad().unwrap()[[0, 0]], 3.0);
        assert_eq!(b.grad().unwrap()[[0, 0]], 2.0);
    }

    #[test]
    fn test_matmul_backward() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[1, 2], true);
        let b = Tensor::from_vec(vec![3.0, 4.0], &[2, 1], true);
        let mut c = matmul(&a, &b);

        c.backward();

        assert!(a.grad().is_some());
        assert!(b.grad().is_some());
    }
}

