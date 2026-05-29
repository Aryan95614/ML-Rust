//! Automatic differentiation primitives

use ndarray::Array2;
use std::sync::Arc;

/// Trait for gradient functions used in backpropagation
pub trait GradFn {
    /// Computes gradients for inputs given upstream gradient
    fn backward(&self, grad: &Array2<f32>) -> Vec<Array2<f32>>;
}

/// Manages gradient storage and computation for a tensor
#[derive(Clone)]
pub(crate) struct GradientManager {
    grad: Option<Array2<f32>>,
    requires_grad: bool,
    grad_fn: Option<Arc<dyn GradFn>>,
}

impl GradientManager {
    pub fn new(requires_grad: bool) -> Self {
        Self {
            grad: None,
            requires_grad,
            grad_fn: None,
        }
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn grad(&self) -> Option<&Array2<f32>> {
        self.grad.as_ref()
    }

    pub fn grad_mut(&mut self) -> Option<&mut Array2<f32>> {
        self.grad.as_mut()
    }

    pub fn set_grad(&mut self, grad: Array2<f32>) {
        self.grad = Some(grad);
    }

    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    pub fn grad_fn(&self) -> Option<&Arc<dyn GradFn>> {
        self.grad_fn.as_ref()
    }

    pub fn set_grad_fn(&mut self, grad_fn: Arc<dyn GradFn>) {
        self.grad_fn = Some(grad_fn);
    }
}

