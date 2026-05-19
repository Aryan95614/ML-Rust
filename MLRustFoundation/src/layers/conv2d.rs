//! Convolutional layer implementation

use crate::tensor::data::Tensor;
use crate::layers::layer::{Layer, ParamVec};
use crate::ops::conv2d::{conv2d, Stride, Padding};
use crate::ops;
use crate::error::Result;
use rand::Rng;
use rand_distr::{Normal, Distribution};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct Conv2D {
    pub kernel: Tensor,
    pub bias: Tensor,
    pub stride: Stride,
    pub padding: Padding,
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel_size: (usize, usize),
}

impl Conv2D {
    pub fn new<R: Rng>(
        input_channels: usize,
        output_channels: usize,
        kernel_size: (usize, usize),
        stride: Stride,
        padding: Padding,
        rng: &mut R,
    ) -> Self {
        let (k_h, k_w) = kernel_size;

        // He initialization for kernel
        let fan_in = input_channels * k_h * k_w;
        let std_dev = (2.0 / fan_in as f32).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();

        let kernel_data = ndarray::Array4::from_shape_fn((output_channels, input_channels, k_h, k_w), |(_, _, _, _)| {
            normal.sample(rng)
        }).into_dyn();
        let kernel = Tensor::new(kernel_data, (output_channels, input_channels, k_h, k_w), true);

        // Bias initialized to zeros
        let bias_data = ndarray::Array4::zeros((1, output_channels, 1, 1)).into_dyn(); // Reshape for broadcasting
        let bias = Tensor::new(bias_data, (1, output_channels, 1, 1), true);

        Conv2D {
            kernel,
            bias,
            stride,
            padding,
            input_channels,
            output_channels,
            kernel_size,
        }
    }
}

impl Layer for Conv2D {
    fn forward(&self, x: &Tensor) -> Tensor {
        let output = conv2d(x, &self.kernel, self.stride, self.padding);
        ops::add(&output, &self.bias)
    }

    fn params(&mut self) -> ParamVec {
        let mut params = ParamVec::new();
        params.push(&mut self.kernel);
        params.push(&mut self.bias);
        params
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

