//! Error types for MLRustFoundation
//!
//! This module defines all error types used throughout the library.

use std::fmt;

/// Result type alias for MLRustFoundation operations
pub type Result<T> = std::result::Result<T, MLError>;

/// Main error type for MLRustFoundation
#[derive(Debug, Clone)]
pub enum MLError {
    /// Shape mismatch between tensors
    ShapeMismatch {
        expected: String,
        got: String,
        operation: String,
    },

    /// Invalid dimensions for operation
    DimensionError {
        expected: usize,
        got: usize,
        operation: String,
    },

    /// Gradient computation error
    GradientError(String),

    /// Lock acquisition failure
    LockError(String),

    /// Invalid tensor operation
    InvalidOperation(String),

    /// Distribution/random number generation error
    DistributionError(String),

    /// I/O error
    IoError(String),

    /// Serialization error
    SerializationError(String),

    /// Data loading error
    DataError(String),

    /// Invalid configuration
    ConfigError(String),
}

impl fmt::Display for MLError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MLError::ShapeMismatch { expected, got, operation } => {
                write!(f, "Shape mismatch in {}: expected {}, got {}", operation, expected, got)
            }
            MLError::DimensionError { expected, got, operation } => {
                write!(f, "Dimension error in {}: expected {} dimensions, got {}", operation, expected, got)
            }
            MLError::GradientError(msg) => write!(f, "Gradient error: {}", msg),
            MLError::LockError(msg) => write!(f, "Lock acquisition failed: {}", msg),
            MLError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            MLError::DistributionError(msg) => write!(f, "Distribution error: {}", msg),
            MLError::IoError(msg) => write!(f, "I/O error: {}", msg),
            MLError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            MLError::DataError(msg) => write!(f, "Data error: {}", msg),
            MLError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for MLError {}

impl From<std::io::Error> for MLError {
    fn from(err: std::io::Error) -> Self {
        MLError::IoError(err.to_string())
    }
}

impl From<bincode::Error> for MLError {
    fn from(err: bincode::Error) -> Self {
        MLError::SerializationError(err.to_string())
    }
}

impl<T> From<std::sync::PoisonError<T>> for MLError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        MLError::LockError(err.to_string())
    }
}

