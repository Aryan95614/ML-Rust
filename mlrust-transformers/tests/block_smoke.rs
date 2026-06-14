//! End-to-end smoke test: embedding → transformer block → finite output.

use mlrust_core::{Layer, Tensor};
use mlrust_transformers::utils::causal_mask;
use mlrust_transformers::{Embedding, TransformerBlock};

#[test]
fn embedding_into_block_produces_finite_output() {
    let vocab = 32;
    let d_model = 16;
    let d_ff = 64;
    let seq = 6;

    let emb = Embedding::new(vocab, d_model);
    let block = TransformerBlock::new(d_model, d_ff);

    let tokens: Vec<usize> = (0..seq).map(|i| i % vocab).collect();
    let x = emb.forward(&tokens);
    let y = block.forward(&x);

    assert_eq!(y.shape(), (seq, d_model));
    for v in y.data().iter() {
        assert!(v.is_finite(), "non-finite output element {v}");
    }
}

#[test]
fn block_handles_causal_mask() {
    let d_model = 8;
    let block = TransformerBlock::new(d_model, 32);
    let x = Tensor::randn((4, d_model), false);
    let mask = causal_mask(4);
    let y = block.forward_with_mask(&x, Some(&mask));
    assert_eq!(y.shape(), (4, d_model));
}

#[test]
fn parameter_count_matches_expected() {
    let mut block = TransformerBlock::new(8, 32);
    // ln1(2) + attn(8) + ln2(2) + ffn(4) = 16
    assert_eq!(block.params().len(), 16);
}
