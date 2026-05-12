use smallvec::SmallVec;
use crate::tensor::data::Tensor;
use std::any::Any;

// A SmallVec is used for efficiency, as most layers will have a small number of parameters.
pub type ParamVec<'a> = SmallVec<[&'a mut Tensor; 8]>;

pub trait Layer: LayerClone + std::fmt::Debug {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn params(&mut self) -> ParamVec;
    fn as_any(&self) -> &dyn Any; // Add as_any method
}

// Trait for cloning boxed `Layer`s
pub trait LayerClone {
    fn clone_box(&self) -> Box<dyn Layer>;
}

impl<T>
    LayerClone for T
where
    T: 'static + Layer + Clone,
{
    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Layer> {
    fn clone(&self) -> Box<dyn Layer> {
        self.clone_box()
    }
}

