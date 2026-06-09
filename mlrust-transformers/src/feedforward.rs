//! Position-wise feed-forward sub-layer: Linear → ReLU → Linear.
//!
//! Standard transformer FFN with hidden width `d_ff` (commonly 4 × d_model).
//! Both projections carry gradients through mlrust-core's matmul/add/relu.

use crate::Linear;
use mlrust_core::{Layer, Tensor};

pub struct FeedForward {
    pub w1: Linear,
    pub w2: Linear,
    pub d_model: usize,
    pub d_ff: usize,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            w1: Linear::new(d_model, d_ff),
            w2: Linear::new(d_ff, d_model),
            d_model,
            d_ff,
        }
    }
}

impl Layer for FeedForward {
    fn forward(&self, input: &Tensor) -> Tensor {
        let h = self.w1.forward(input).relu();
        self.w2.forward(&h)
    }

    fn params(&mut self) -> Vec<&mut Tensor> {
        let mut p = self.w1.params();
        p.extend(self.w2.params());
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_shape_matches_d_model() {
        let ffn = FeedForward::new(16, 64);
        let x = Tensor::randn((5, 16), false);
        let y = ffn.forward(&x);
        assert_eq!(y.shape(), (5, 16));
    }

    #[test]
    fn parameter_count_is_four() {
        // w1.weight, w1.bias, w2.weight, w2.bias
        let mut ffn = FeedForward::new(8, 32);
        assert_eq!(ffn.params().len(), 4);
    }

    #[test]
    fn relu_zeros_negative_activations() {
        // After w1 + relu, all values feeding w2 must be >= 0. Verify via
        // a hand-rolled forward up to the relu boundary.
        let ffn = FeedForward::new(4, 8);
        let x = Tensor::randn((3, 4), false);
        let h = ffn.w1.forward(&x).relu();
        for v in h.data().iter() {
            assert!(*v >= 0.0);
        }
    }
}
