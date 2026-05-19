//! ML foundation library in Rust.
//!
//! Tensor with reverse-mode autograd, a few neural network layers,
//! SGD with momentum, a small training loop, and MNIST examples.

pub mod data;
pub mod error;
pub mod layers;
pub mod loss;
pub mod model;
pub mod ops;
pub mod optim;
pub mod tensor;
pub mod train;

pub use error::{MLError, Result};
