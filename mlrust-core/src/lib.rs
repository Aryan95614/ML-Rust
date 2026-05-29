//! # mlrust-core
//!
//! Core tensor operations and automatic differentiation for the MLRust ecosystem.
//!
//! This library provides the minimal foundation that all MLRust libraries build upon:
//! - Tensor type with reference-counted data sharing
//! - Automatic differentiation via computational graph
//! - Basic operations (add, mul, matmul, activations)
//! - Traits for layers and optimizers
//!
//! # Example
//!
//! ```
//! use mlrust_core::Tensor;
//!
//! let a = Tensor::ones((2, 2), true);
//! let b = Tensor::ones((2, 2), true);
//! let c = a.add(&b);
//! let mut loss = c.sum();
//! loss.backward();
//!
//! // Access gradients
//! println!("Gradient: {:?}", a.grad());
//! ```

pub mod tensor;
pub mod autograd;
pub mod ops;
pub mod traits;
pub mod loss;

// Re-export core types
pub use tensor::Tensor;
pub use autograd::GradFn;
pub use traits::{Layer, Optimizer, Loss};
pub use loss::{mse_loss, bce_loss};

/// Common shape type: (rows, cols)
pub type Shape = (usize, usize);

