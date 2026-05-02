//! Automatic differentiation engine
//!
//! This module implements reverse-mode automatic differentiation (backpropagation)
//! for computing gradients of neural network operations.

use std::sync::{Arc, Mutex};
use ndarray::ArrayD;
use crate::tensor::data::{Tensor, Shape};
use crate::error::{MLError, Result};

/// Gradient function representing an operation in the computational graph
///
/// Each variant stores the inputs needed to compute gradients during
/// the backward pass.
#[derive(Clone, Debug)]
pub enum GradFn {
    Add {
        input1: Tensor,
        input2: Tensor,
    },
    Mul {
        input1: Tensor,
        input2: Tensor,
        output: Tensor,
    },
    MatMul {
        input1: Tensor,
        input2: Tensor,
    },
    ReLU {
        input: Tensor,
        output: Tensor, // Store output to compute gradient
    },
    Softmax {
        output: Tensor, // Store output to compute gradient
    },
    Sigmoid {
        output: Tensor, // Store output to compute gradient
    },
    Conv2D {
        input: Tensor,
        kernel: Tensor,
        stride: crate::ops::conv2d::Stride,
        padding: crate::ops::conv2d::Padding,
    },
    MaxPool2D {
        input: Tensor,
        input_shape: Shape, // Original input shape
        max_indices: ArrayD<usize>, // Indices of max values
    },
    CrossEntropyLoss {
        logits: Tensor,
        targets: Tensor,
    },
    Flatten {
        input: Tensor,
        input_shape: Shape,
    },
    // Placeholder for other operations
    None,
}

