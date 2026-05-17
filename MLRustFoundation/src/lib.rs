//! ML foundation library in Rust.

pub mod error;
pub mod layers;
pub mod loss;
pub mod model;
pub mod ops;
pub mod optim;
pub mod tensor;
pub mod train;

pub use error::{MLError, Result};
