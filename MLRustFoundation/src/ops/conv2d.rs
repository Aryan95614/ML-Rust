//! 2D Convolution operations with im2col/col2im implementation
//!
//! This module implements efficient 2D convolution using the im2col (image to column)
//! algorithm, which transforms the convolution into a matrix multiplication.

use ndarray::{ArrayD, IxDyn, s};
use crate::tensor::data::{Tensor, Shape};
use crate::tensor::autograd::GradFn;
use serde::{Serialize, Deserialize};

/// Stride for convolution operations (height, width)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Stride(pub usize, pub usize);

/// Padding for convolution operations (height, width)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Padding(pub usize, pub usize);

/// Converts image patches to columns for efficient convolution
///
/// The im2col operation transforms a 4D input tensor into a 2D matrix where each row
/// contains a flattened image patch. This allows convolution to be computed as a
/// matrix multiplication.
///
/// # Arguments
///
/// * `input` - Input tensor of shape (N, C, H, W)
/// * `kernel_h` - Kernel height
/// * `kernel_w` - Kernel width
/// * `stride_h` - Vertical stride
/// * `stride_w` - Horizontal stride
/// * `pad_h` - Vertical padding
/// * `pad_w` - Horizontal padding
///
/// # Returns
///
/// A 2D array of shape (N × OH × OW, C × KH × KW) where:
/// - N is batch size
/// - OH, OW are output height and width
/// - C is number of channels
/// - KH, KW are kernel height and width
pub fn im2col(
    input: &ArrayD<f32>,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> ArrayD<f32> {
    let (n, c, h, w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );

    let output_h = (h + 2 * pad_h - kernel_h) / stride_h + 1;
    let output_w = (w + 2 * pad_w - kernel_w) / stride_w + 1;

    let mut cols = ArrayD::zeros(IxDyn(&[
        n * output_h * output_w,
        c * kernel_h * kernel_w,
    ]));

    // Pad the input
    let mut padded_input = ArrayD::zeros(IxDyn(&[n, c, h + 2 * pad_h, w + 2 * pad_w]));
    padded_input
        .slice_mut(s![.., .., pad_h..h + pad_h, pad_w..w + pad_w])
        .assign(input);

    for n_idx in 0..n {
        for out_h_idx in 0..output_h {
            for out_w_idx in 0..output_w {
                let h_start = out_h_idx * stride_h;
                let w_start = out_w_idx * stride_w;

                let patch = padded_input.slice(s![
                    n_idx,
                    ..,
                    h_start..h_start + kernel_h,
                    w_start..w_start + kernel_w,
                ]);

                let col_idx = n_idx * output_h * output_w + out_h_idx * output_w + out_w_idx;
                cols.slice_mut(s![col_idx, ..])
                    .assign(&patch.into_shape((c * kernel_h * kernel_w,)).unwrap());
            }
        }
    }
    cols
}

// Helper function for col2im (backward pass for im2col)
pub fn col2im(
    cols: &ArrayD<f32>,
    input_shape: &[usize], // (N, C, H, W)
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> ArrayD<f32> {
    let (n, c, h, w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );

    let output_h = (h + 2 * pad_h - kernel_h) / stride_h + 1;
    let output_w = (w + 2 * pad_w - kernel_w) / stride_w + 1;

    let mut padded_grad_input = ArrayD::zeros(IxDyn(&[n, c, h + 2 * pad_h, w + 2 * pad_w]));

    for n_idx in 0..n {
        for out_h_idx in 0..output_h {
            for out_w_idx in 0..output_w {
                let h_start = out_h_idx * stride_h;
                let w_start = out_w_idx * stride_w;

                let col_idx = n_idx * output_h * output_w + out_h_idx * output_w + out_w_idx;
                let col_patch = cols.slice(s![col_idx, ..]);

                padded_grad_input
                    .slice_mut(s![
                        n_idx,
                        ..,
                        h_start..h_start + kernel_h,
                        w_start..w_start + kernel_w,
                    ])
                    .assign(&col_patch.into_shape((c, kernel_h, kernel_w)).unwrap());
            }
        }
    }

    // Unpad the result
    padded_grad_input
        .slice(s![.., .., pad_h..h + pad_h, pad_w..w + pad_w])
        .to_owned()
        .into_dyn()
}

pub fn conv2d(
    input: &Tensor,
    kernel: &Tensor,
    stride: Stride,
    padding: Padding,
) -> Tensor {
    let input_data = input.get_data();
    let kernel_data = kernel.get_data();

    // Input shape: (N, C_in, H_in, W_in)
    // Kernel shape: (C_out, C_in, K_h, K_w)
    assert_eq!(input_data.ndim(), 4, "Input must be 4D (N, C, H, W)");
    assert_eq!(kernel_data.ndim(), 4, "Kernel must be 4D (C_out, C_in, K_h, K_w)");
    assert_eq!(input_data.shape()[1], kernel_data.shape()[1], "Input channels must match kernel input channels");

    let (n, c_in, h_in, w_in) = (
        input_data.shape()[0],
        input_data.shape()[1],
        input_data.shape()[2],
        input_data.shape()[3],
    );
    let (c_out, _, k_h, k_w) = (
        kernel_data.shape()[0],
        kernel_data.shape()[1],
        kernel_data.shape()[2],
        kernel_data.shape()[3],
    );

    let output_h = (h_in + 2 * padding.0 - k_h) / stride.0 + 1;
    let output_w = (w_in + 2 * padding.1 - k_w) / stride.1 + 1;

    let cols = im2col(&input_data, k_h, k_w, stride.0, stride.1, padding.0, padding.1);

    // Reshape kernel for matrix multiplication: (C_out, C_in * K_h * K_w)
    let kernel_reshaped = kernel_data
        .into_shape((c_out, c_in * k_h * k_w))
        .unwrap();

    // Perform matrix multiplication: (N * O_h * O_w, C_in * K_h * K_w) @ (C_in * K_h * K_w, C_out).T
    // Result: (N * O_h * O_w, C_out)
    let cols_2d = cols.into_dimensionality::<ndarray::Ix2>().unwrap();
    let kernel_2d = kernel_reshaped.into_dimensionality::<ndarray::Ix2>().unwrap();
    let output_cols = cols_2d.dot(&kernel_2d.t()).into_dyn();

    // Reshape output back to (N, C_out, O_h, O_w)
    let output_data = output_cols
        .into_shape((n, output_h, output_w, c_out))
        .unwrap()
        .permuted_axes([0, 3, 1, 2]) // Permute to NCHW
        .to_owned()
        .into_dyn();

    let output_shape = Shape(vec![n, c_out, output_h, output_w]);
    let requires_grad = input.requires_grad || kernel.requires_grad;

    let mut output = Tensor::new(output_data, output_shape, requires_grad);

    if requires_grad {
        output.grad_fn = Some(Box::new(GradFn::Conv2D {
            input: input.clone(),
            kernel: kernel.clone(),
            stride,
            padding,
        }));
    }

    output
}