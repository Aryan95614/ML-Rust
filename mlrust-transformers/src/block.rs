//! Pre-LN transformer encoder block.
//!
//! ```text
//! h = x + SelfAttention(LayerNorm(x))
//! y = h + FeedForward(LayerNorm(h))
//! ```
//!
//! Pre-LN is preferred over Post-LN for stable training without warmup
//! (Xiong et al. 2020).

use crate::{FeedForward, LayerNorm, SelfAttention};
use mlrust_core::{Layer, Tensor};
use ndarray::Array2;

pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub attn: SelfAttention,
    pub ln2: LayerNorm,
    pub ffn: FeedForward,
    pub d_model: usize,
}

impl TransformerBlock {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            ln1: LayerNorm::new(d_model),
            attn: SelfAttention::new(d_model),
            ln2: LayerNorm::new(d_model),
            ffn: FeedForward::new(d_model, d_ff),
            d_model,
        }
    }

    pub fn forward_with_mask(&self, input: &Tensor, mask: Option<&Array2<f32>>) -> Tensor {
        let h = input.add(&self.attn.forward_with_mask(&self.ln1.forward(input), mask));
        h.add(&self.ffn.forward(&self.ln2.forward(&h)))
    }
}

impl Layer for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_with_mask(input, None)
    }

    fn params(&mut self) -> Vec<&mut Tensor> {
        let mut p = vec![&mut self.ln1.gamma, &mut self.ln1.beta];
        p.extend(self.attn.params());
        p.push(&mut self.ln2.gamma);
        p.push(&mut self.ln2.beta);
        p.extend(self.ffn.params());
        p
    }
}

