//! Core traits for layers and optimizers

use crate::Tensor;

/// Trait for neural network layers
///
/// Layers transform inputs to outputs and expose trainable parameters.
pub trait Layer {
    /// Forward pass: transforms input to output
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Returns mutable references to trainable parameters
    ///
    /// Note: Returns Vec instead of smallvec for simplicity in core
    fn params(&mut self) -> Vec<&mut Tensor>;
}

/// Trait for optimizers
///
/// Optimizers update parameters based on their gradients.
pub trait Optimizer {
    /// Updates parameters using their gradients
    fn step(&self, params: &mut [&mut Tensor]);

    /// Zeros out gradients for all parameters
    fn zero_grad(&self, params: &mut [&mut Tensor]) {
        for param in params.iter_mut() {
            param.zero_grad();
        }
    }

    /// Returns current learning rate
    fn lr(&self) -> f32;

    /// Sets learning rate (for schedulers)
    fn set_lr(&mut self, lr: f32);
}

/// Trait for loss functions
///
/// Loss functions compute a scalar loss from predictions and targets.
pub trait Loss {
    /// Computes loss between predictions and targets
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Tensor;
}

