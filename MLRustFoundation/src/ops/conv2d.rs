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

