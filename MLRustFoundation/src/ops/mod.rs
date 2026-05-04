//! Core tensor operations and transformations
//!
//! This module provides fundamental operations for tensor computation including:
//! - Arithmetic operations (add, mul, matmul)
//! - Activation functions (ReLU, Softmax, Sigmoid)
//! - Convolutional operations (conv2d)
//!
//! All operations support automatic differentiation and gradient computation.

pub mod activations;
pub mod add;
pub mod conv2d;
pub mod matmul;
pub mod mul;

// Re-export commonly used functions
pub use add::add;
pub use matmul::matmul;
pub use mul::mul;
