use crate::tensor::data::{Tensor, Shape};
use crate::layers::layer::{Layer, ParamVec};
use ndarray::IxDyn;
use std::any::Any;

#[derive(Debug, Clone)]
pub struct Flatten;

impl Flatten {
    pub fn new() -> Self {
        Flatten {}
    }
}

impl Layer for Flatten {
    fn forward(&self, x: &Tensor) -> Tensor {
        let input_data = x.get_data();
        let (n, c, h, w) = (
            input_data.shape()[0],
            input_data.shape()[1],
            input_data.shape()[2],
            input_data.shape()[3],
        );

        let output_size = c * h * w;
        let output_data = input_data.into_shape((n, output_size)).unwrap().to_owned().into_dyn();
        let output_shape = Shape(vec![n, output_size]);

        let requires_grad = x.requires_grad;

        let mut output = Tensor::new(output_data, output_shape, requires_grad);

        if requires_grad {
            output.grad_fn = Some(Box::new(crate::tensor::autograd::GradFn::Flatten {
                input: x.clone(),
                input_shape: x.shape.clone(),
            }));
        }

        output
    }

    fn params(&mut self) -> ParamVec {
        ParamVec::new() // No parameters for Flatten
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

