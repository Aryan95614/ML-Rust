use std::ops::Shr;
use std::any::Any;
use crate::tensor::data::Tensor;
use crate::layers::layer::{Layer, ParamVec};
use crate::layers::input::Input;
// Add other layers as they are implemented

#[derive(Debug, Clone)]
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    pub fn add<L: Layer + 'static>(mut self, layer: L) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn get_layers(&self) -> &Vec<Box<dyn Layer>> {
        &self.layers
    }
}

impl Layer for Sequential {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut current = x.clone();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }

    fn params(&mut self) -> ParamVec {
        let mut all_params = ParamVec::new();
        for layer in &mut self.layers {
            all_params.extend(layer.params());
        }
        all_params
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Implement >> operator for Input >> Layer
impl<L> Shr<L> for Input
where
    L: Layer + 'static,
{
    type Output = Sequential;

    fn shr(self, rhs: L) -> Self::Output {
        let mut seq = Sequential::new();
        // The Input layer itself is not added to the sequential model,
        // it just defines the starting shape.
        // The first actual processing layer is `rhs`.
        seq.add(rhs)
    }
}

// Implement >> operator for Sequential >> Layer
impl<L> Shr<L> for Sequential
where
    L: Layer + 'static,
{
    type Output = Sequential;

    fn shr(self, rhs: L) -> Self::Output {
        self.add(rhs)
    }
}
