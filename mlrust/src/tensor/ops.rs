use ndarray::Array2;
use rand_distr::{Distribution, StandardNormal};

use super::Shape;

pub(crate) struct TensorFactory;

impl TensorFactory {
    
    pub fn zeros(shape: Shape) -> Array2<f32> {
        Array2::zeros(shape)
    }
    
    pub fn ones(shape: Shape) -> Array2<f32> {
        Array2::ones(shape)
    }

    pub fn randn(shape: Shape) -> Array2<f32> {
        let mut rng = rand::rng();
        let normal = StandardNormal;
        Array2::from_shape_fn(shape, |_| normal.sample(&mut rng))
    }
}
