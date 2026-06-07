//! Scaled dot-product attention.
//!
//! `Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V`
//!
//! Forward-only: softmax has no autograd path in mlrust-core yet, so the
//! returned tensor is detached from the computation graph.

use crate::utils::softmax_rows;
use mlrust_core::Tensor;
use ndarray::Array2;

/// Compute scaled dot-product attention.
///
/// Shapes: `q (seq_q, d_k)`, `k (seq_k, d_k)`, `v (seq_k, d_v)` →
/// `out (seq_q, d_v)`. The optional mask is added to the scores before
/// softmax (use `f32::NEG_INFINITY` to block positions).
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Array2<f32>>,
) -> Tensor {
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();

    let d_k = q_data.ncols();
    let scale = 1.0_f32 / (d_k as f32).sqrt();

    let mut scores = q_data.dot(&k_data.t()) * scale;
    if let Some(m) = mask {
        assert_eq!(scores.dim(), m.dim(), "mask shape must match scores");
        scores = scores + m;
    }

    let attn = softmax_rows(&scores);
    Tensor::new(attn.dot(&v_data), false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_shape() {
        let q = Tensor::randn((3, 8), false);
        let k = Tensor::randn((5, 8), false);
        let v = Tensor::randn((5, 4), false);
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), (3, 4));
    }

    #[test]
    fn causal_mask_blocks_future_positions() {
        // With a causal mask, position 0 can only attend to position 0,
        // so out.row(0) must equal v.row(0) regardless of later v rows.
        use crate::utils::causal_mask;

        let q = Tensor::randn((4, 6), false);
        let k = Tensor::randn((4, 6), false);
        let v_data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, // row 0
                10.0, 20.0, 30.0, 100.0, 200.0, 300.0, 1000.0, 2000.0, 3000.0,
            ],
        )
        .unwrap();
        let expected_row0 = v_data.row(0).to_owned();
        let v = Tensor::new(v_data, false);

        let mask = causal_mask(4);
        let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask)).data();
        for (a, b) in out.row(0).iter().zip(expected_row0.iter()) {
            assert!((a - b).abs() < 1e-5, "got {a} expected {b}");
        }
    }

    #[test]
    fn uniform_keys_yield_mean_of_values() {
        // If all key rows are identical, every query gets the same softmax
        // (uniform), so output rows = mean of value rows.
        let q = Tensor::randn((2, 4), false);
        let k = Tensor::ones((6, 4), false);
        let v_data =
            Array2::from_shape_vec((6, 3), (0..18).map(|i| i as f32).collect()).unwrap();
        let v_mean = v_data.mean_axis(ndarray::Axis(0)).unwrap();
        let v = Tensor::new(v_data, false);

        let out = scaled_dot_product_attention(&q, &k, &v, None).data();
        for r in out.axis_iter(ndarray::Axis(0)) {
            for (a, b) in r.iter().zip(v_mean.iter()) {
                assert!((a - b).abs() < 1e-5, "got {a} expected {b}");
            }
        }
    }
}
