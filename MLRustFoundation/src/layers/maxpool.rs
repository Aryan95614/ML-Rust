use crate::tensor::data::{Tensor, Shape};
use crate::layers::layer::{Layer, ParamVec};
use ndarray::{ArrayD, IxDyn, s};
use crate::tensor::autograd::GradFn;
use std::any::Any;

#[derive(Debug, Clone)]
pub struct MaxPool2D {
    pub kernel_size: (usize, usize), // (height, width)
    // Store input shape and max indices for backward pass
    input_shape: Option<Shape>,
    max_indices: Option<ArrayD<usize>>, // Store indices of max values
}

impl MaxPool2D {
    pub fn new(kernel_size: (usize, usize)) -> Self {
        MaxPool2D {
            kernel_size,
            input_shape: None,
            max_indices: None,
        }
    }

    fn max_pool_forward(
        &mut self,
        input: &Tensor,
        kernel_h: usize,
        kernel_w: usize,
    ) -> (ArrayD<f32>, ArrayD<usize>) {
        let input_data = input.get_data();
        let (n, c, h_in, w_in) = (
            input_data.shape()[0],
            input_data.shape()[1],
            input_data.shape()[2],
            input_data.shape()[3],
        );

        let output_h = h_in / kernel_h;
        let output_w = w_in / kernel_w;

        let mut output_data = ArrayD::zeros(IxDyn(&[n, c, output_h, output_w]));
        let mut max_indices = ArrayD::zeros(IxDyn(&[n, c, output_h, output_w])); // Store flat index

        for n_idx in 0..n {
            for c_idx in 0..c {
                for h_out_idx in 0..output_h {
                    for w_out_idx in 0..output_w {
                        let h_start = h_out_idx * kernel_h;
                        let w_start = w_out_idx * kernel_w;

                        let patch = input_data.slice(s![
                            n_idx,
                            c_idx,
                            h_start..h_start + kernel_h,
                            w_start..w_start + kernel_w,
                        ]);

                        let (max_val, flat_idx) = patch.iter().enumerate().fold(
                            (f32::NEG_INFINITY, 0),
                            |(val_max, idx_max), (idx, &val)| {
                                if val > val_max {
                                    (val, idx)
                                } else {
                                    (val_max, idx_max)
                                }
                            },
                        );
                        output_data[[n_idx, c_idx, h_out_idx, w_out_idx]] = max_val;
                        max_indices[[n_idx, c_idx, h_out_idx, w_out_idx]] = flat_idx;
                    }
                }
            }
        }
        (output_data, max_indices)
    }
}

impl Layer for MaxPool2D {
    fn forward(&self, x: &Tensor) -> Tensor {
        let (kernel_h, kernel_w) = self.kernel_size;
        let mut cloned_self = self.clone(); // Clone to modify input_shape and max_indices

        let (output_data, max_indices) = cloned_self.max_pool_forward(x, kernel_h, kernel_w);

        cloned_self.input_shape = Some(x.shape.clone());
        cloned_self.max_indices = Some(max_indices);

        let output_shape = Shape(output_data.shape().to_vec());
        let requires_grad = x.requires_grad;

        let mut output = Tensor::new(output_data, output_shape, requires_grad);

        if requires_grad {
            output.grad_fn = Some(Box::new(GradFn::MaxPool2D {
                input: x.clone(),
                input_shape: x.shape.clone(), // Store input shape for backward pass
                max_indices: cloned_self.max_indices.clone().unwrap(), // Store max_indices for backward pass
            }));
        }

        output
    }

    fn params(&mut self) -> ParamVec {
        ParamVec::new() // No parameters for MaxPool2D
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

