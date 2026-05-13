use crate::tensor::data::Tensor;
use crate::layers::layer::{Layer, ParamVec};
use crate::ops::activations;
use std::any::Any;

#[derive(Debug, Clone)]
pub struct ReLU; // ReLU doesn't have parameters, so it's a zero-sized struct

impl ReLU {
    pub fn new() -> Self {
        ReLU {}
    }
}

impl Layer for ReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        activations::relu(x)
    }

    fn params(&mut self) -> ParamVec {
        ParamVec::new() // No parameters for ReLU
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug, Clone)]
pub struct Softmax; // Softmax doesn't have parameters

impl Softmax {
    pub fn new() -> Self {
        Softmax {}
    }
}

impl Layer for Softmax {
    fn forward(&self, x: &Tensor) -> Tensor {
        activations::softmax(x)
    }

    fn params(&mut self) -> ParamVec {
        ParamVec::new() // No parameters for Softmax
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}