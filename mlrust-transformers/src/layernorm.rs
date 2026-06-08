//! Layer normalization with learnable per-feature scale and shift.
//!
//! `y = (x - mean(x, axis=-1)) / sqrt(var(x, axis=-1) + eps) * gamma + beta`
//!
//! Forward-only — mean/var have no autograd path in mlrust-core, so the
//! returned tensor is detached. Hooked up later when the core gets
//! reduction backwards along an axis.

use mlrust_core::Tensor;
use ndarray::{Array2, Axis};

pub struct LayerNorm {
    pub gamma: Tensor, // (1, d_model)
    pub beta: Tensor,  // (1, d_model)
    pub d_model: usize,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(d_model: usize) -> Self {
        Self {
            gamma: Tensor::new(Array2::ones((1, d_model)), true),
            beta: Tensor::new(Array2::zeros((1, d_model)), true),
            d_model,
            eps: 1e-5,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let x = input.data();
        assert_eq!(x.ncols(), self.d_model, "input width must equal d_model");

        let mean = x.mean_axis(Axis(1)).unwrap();
        let mut centered = Array2::<f32>::zeros(x.raw_dim());
        for (i, m) in mean.iter().enumerate() {
            for j in 0..self.d_model {
                centered[[i, j]] = x[[i, j]] - m;
            }
        }

        let var = centered.mapv(|v| v * v).mean_axis(Axis(1)).unwrap();
        let mut out = Array2::<f32>::zeros(x.raw_dim());
        let gamma = self.gamma.data();
        let beta = self.beta.data();
        for i in 0..x.nrows() {
            let denom = (var[i] + self.eps).sqrt();
            for j in 0..self.d_model {
                out[[i, j]] = centered[[i, j]] / denom * gamma[[0, j]] + beta[[0, j]];
            }
        }
        Tensor::new(out, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row_stats(arr: &Array2<f32>, row: usize) -> (f32, f32) {
        let r = arr.row(row);
        let mean = r.iter().sum::<f32>() / r.len() as f32;
        let var = r.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / r.len() as f32;
        (mean, var)
    }

    #[test]
    fn rows_normalized_to_zero_mean_unit_var() {
        let ln = LayerNorm::new(8);
        let x = Tensor::randn((4, 8), false);
        let y = ln.forward(&x);
        let yd = y.data();
        for i in 0..4 {
            let (mean, var) = row_stats(&yd, i);
            assert!(mean.abs() < 1e-4, "row {i} mean={mean}");
            assert!((var - 1.0).abs() < 1e-2, "row {i} var={var}");
        }
    }

    #[test]
    fn beta_shifts_output_uniformly() {
        let mut ln = LayerNorm::new(4);
        ln.beta.set_data(Array2::from_elem((1, 4), 3.0));
        let x = Tensor::randn((2, 4), false);
        let y = ln.forward(&x);
        let yd = y.data();
        for i in 0..2 {
            let (mean, _) = row_stats(&yd, i);
            assert!((mean - 3.0).abs() < 1e-4);
        }
    }

    #[test]
    fn handles_constant_input_without_nan() {
        let ln = LayerNorm::new(4);
        let x = Tensor::ones((2, 4), false);
        let y = ln.forward(&x);
        for v in y.data().iter() {
            assert!(v.is_finite(), "got non-finite {v}");
        }
    }
}
