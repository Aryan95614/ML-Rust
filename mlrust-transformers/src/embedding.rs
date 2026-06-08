//! Token embedding table.
//!
//! Maps a sequence of integer token IDs to a dense `(seq_len, d_model)` tensor
//! by row-gather from a learnable `(vocab_size, d_model)` weight matrix.

use mlrust_core::Tensor;
use ndarray::Array2;
use rand_distr::{Distribution, Normal};

pub struct Embedding {
    pub weight: Tensor,
    pub vocab_size: usize,
    pub d_model: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        let std = 0.02_f32; // GPT-style small init
        let normal = Normal::new(0.0, std).expect("valid normal");
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..vocab_size * d_model)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let weight_arr = Array2::from_shape_vec((vocab_size, d_model), data).expect("shape ok");
        Self {
            weight: Tensor::new(weight_arr, true),
            vocab_size,
            d_model,
        }
    }

    /// Gather embedding rows for the given token IDs.
    ///
    /// Panics if any token ID is out of `[0, vocab_size)`.
    pub fn forward(&self, tokens: &[usize]) -> Tensor {
        let w = self.weight.data();
        let mut out = Array2::<f32>::zeros((tokens.len(), self.d_model));
        for (i, &tok) in tokens.iter().enumerate() {
            assert!(tok < self.vocab_size, "token id {tok} out of vocab");
            out.row_mut(i).assign(&w.row(tok));
        }
        Tensor::new(out, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_shape_matches_seq_and_model() {
        let emb = Embedding::new(100, 16);
        let out = emb.forward(&[3, 7, 42, 0]);
        assert_eq!(out.shape(), (4, 16));
    }

    #[test]
    fn same_token_yields_same_row() {
        let emb = Embedding::new(10, 8);
        let out = emb.forward(&[5, 5, 5]);
        let r0 = out.data().row(0).to_owned();
        let r1 = out.data().row(1).to_owned();
        let r2 = out.data().row(2).to_owned();
        assert_eq!(r0, r1);
        assert_eq!(r1, r2);
    }

    #[test]
    #[should_panic]
    fn out_of_vocab_panics() {
        let emb = Embedding::new(4, 2);
        let _ = emb.forward(&[10]);
    }
}
