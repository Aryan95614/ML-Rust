//! Forward-pass tensor helpers that mlrust-core does not expose yet.
//!
//! These work on `ndarray::Array2<f32>` directly. Once mlrust-core grows
//! gradient-tracked exp/div, the softmax here will be replaced by an
//! autograd-aware version.

use ndarray::{Array2, Axis};

/// Causal (lower-triangular) attention mask for a sequence of length `n`.
///
/// Returns an `(n, n)` matrix that is `0` on and below the diagonal and
/// `-inf` above it. Add this to attention scores before softmax to prevent
/// each position from attending to future positions.
pub fn causal_mask(n: usize) -> Array2<f32> {
    let mut m = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            m[[i, j]] = f32::NEG_INFINITY;
        }
    }
    m
}

/// Row-wise softmax with the standard max-subtraction trick for stability.
pub fn softmax_rows(x: &Array2<f32>) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros(x.raw_dim());
    for (i, row) in x.axis_iter(Axis(0)).enumerate() {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (j, e) in exps.iter().enumerate() {
            out[[i, j]] = e / sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_sum_to_one() {
        let x = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0])
            .unwrap();
        let p = softmax_rows(&x);
        for row in p.axis_iter(Axis(0)) {
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row sum {s}");
        }
    }

    #[test]
    fn handles_large_values_without_overflow() {
        let x = Array2::from_shape_vec((1, 3), vec![1000.0, 1001.0, 1002.0]).unwrap();
        let p = softmax_rows(&x);
        let s: f32 = p.row(0).iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        assert!(p[[0, 2]] > p[[0, 0]]);
    }

    #[test]
    fn uniform_input_yields_uniform_output() {
        let x = Array2::from_elem((1, 5), 0.7);
        let p = softmax_rows(&x);
        for v in p.row(0).iter() {
            assert!((v - 0.2).abs() < 1e-6);
        }
    }
}
