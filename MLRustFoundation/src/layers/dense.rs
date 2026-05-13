use crate::tensor::data::{Tensor, Shape};
use crate::layers::layer::{Layer, ParamVec};
use crate::ops;
use rand::{thread_rng, Rng};
use rand_distr::{Normal, Distribution};
use ndarray::ArrayD;
use std::any::Any;

#[derive(Debug, Clone)]
pub struct Dense {
    pub weights: Tensor,
    pub bias: Tensor,
    pub input_size: usize,
    pub output_size: usize,
}

impl Dense {
    pub fn new<R: Rng>(input_size: usize, output_size: usize, rng: &mut R) -> Self {
        // He init
        let std_dev = (2.0 / input_size as f32).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();

        let weights_data = ndarray::Array2::from_shape_fn((input_size, output_size), |(_, _)| {
            normal.sample(rng)
        }).into_dyn();
        let weights = Tensor::new(weights_data, (input_size, output_size), true);

        // Bias initialized to zeros
        let bias_data = ndarray::Array2::zeros((1, output_size)).into_dyn();
        let bias = Tensor::new(bias_data, (1, output_size), true);

        Dense {
            weights,
            bias,
            input_size,
            output_size,
        }
    }
}

impl Layer for Dense {
    fn forward(&self, x: &Tensor) -> Tensor {
        // x @ W + b
        let output = ops::matmul(x, &self.weights);
        ops::add(&output, &self.bias)
    }

    fn params(&mut self) -> ParamVec {
        let mut params = ParamVec::new();
        params.push(&mut self.weights);
        params.push(&mut self.bias);
        params
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

