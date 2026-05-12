//! Neural network layer implementations
//!
//! This module provides a collection of composable neural network layers:
//!
//! - **Dense**: Fully connected layer with learnable weights and biases
//! - **Conv2D**: 2D convolutional layer for image processing
//! - **MaxPool2D**: Max pooling layer for downsampling
//! - **Flatten**: Reshapes multi-dimensional tensors to 1D
//! - **Activations**: ReLU, Softmax, Sigmoid activation layers
//! - **Input**: Entry point layer defining input shape
//! - **Sequential**: Chain layers together with the `>>` operator
//!
//! # Example
//!
//! ```no_run
//! use mlrust::layers::*;
//!
//! let model = Input::image(1, 28, 28)
//!     >> Dense::new(784, 128, &mut rng).unwrap()
//!     >> ReLU::new()
//!     >> Dense::new(128, 10, &mut rng).unwrap()
//!     >> Softmax::new();
//! ```

pub mod activations;
pub mod builder;
pub mod conv2d;
pub mod dense;
pub mod flatten;
pub mod input;
pub mod layer;
pub mod maxpool;

