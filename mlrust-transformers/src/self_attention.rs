//! Single-head self-attention block.
//!
//! Wraps four `Linear` projections (Q, K, V, output) around the
//! [`crate::attention::scaled_dot_product_attention`] kernel.
//!
//! Forward-only end-to-end (the softmax inside `scaled_dot_product_attention`
//! detaches the result from the autograd graph).

use crate::attention::scaled_dot_product_attention;
use crate::Linear;
use mlrust_core::{Layer, Tensor};
use ndarray::Array2;

pub struct SelfAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub d_model: usize,
}

impl SelfAttention {
    pub fn new(d_model: usize) -> Self {
        Self {
            q_proj: Linear::new(d_model, d_model),
            k_proj: Linear::new(d_model, d_model),
            v_proj: Linear::new(d_model, d_model),
            out_proj: Linear::new(d_model, d_model),
            d_model,
        }
    }

    pub fn forward_with_mask(&self, input: &Tensor, mask: Option<&Array2<f32>>) -> Tensor {
        let q = self.q_proj.forward(input);
        let k = self.k_proj.forward(input);
        let v = self.v_proj.forward(input);
        let attended = scaled_dot_product_attention(&q, &k, &v, mask);
        self.out_proj.forward(&attended)
    }
}

impl Layer for SelfAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_with_mask(input, None)
    }

    fn params(&mut self) -> Vec<&mut Tensor> {
        let mut p = self.q_proj.params();
        p.extend(self.k_proj.params());
        p.extend(self.v_proj.params());
        p.extend(self.out_proj.params());
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_shape_preserved() {
        let sa = SelfAttention::new(16);
        let x = Tensor::randn((6, 16), false);
        let y = sa.forward(&x);
        assert_eq!(y.shape(), (6, 16));
    }

    #[test]
    fn parameter_count_is_eight() {
        // 4 Linear layers × (weight + bias) = 8 params
        let mut sa = SelfAttention::new(8);
        assert_eq!(sa.params().len(), 8);
    }

    #[test]
    fn causal_mask_is_accepted() {
        use crate::utils::causal_mask;
        let sa = SelfAttention::new(8);
        let x = Tensor::randn((5, 8), false);
        let mask = causal_mask(5);
        let y = sa.forward_with_mask(&x, Some(&mask));
        assert_eq!(y.shape(), (5, 8));
    }
}
