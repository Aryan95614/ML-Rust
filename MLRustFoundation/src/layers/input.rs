use crate::tensor::data::{Tensor, Shape};
use crate::layers::layer::{Layer, ParamVec};
use std::any::Any;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Input {
    pub shape: Shape,
}

impl Input {
    pub fn new<S>(shape: S) -> Self
    where
        S: Into<Shape>,
    {
        Input { shape: shape.into() }
    }

    // Helper for image input (N, C, H, W)
    pub fn image(c: usize, h: usize, w: usize) -> Self {
        Input { shape: Shape(vec![1, c, h, w]) } // Batch size 1 for now, will be adjusted by data loader
    }
}

impl Layer for Input {
    fn forward(&self, x: &Tensor) -> Tensor {
        // Input layer just passes the tensor through, potentially validating shape
        // For now, we assume the input tensor already has the correct shape.
        x.clone()
    }

    fn params(&mut self) -> ParamVec {
        ParamVec::new() // No parameters for Input layer
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}