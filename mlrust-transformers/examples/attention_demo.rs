//! Tiny end-to-end demo: embed a 6-token sequence, run it through one
//! transformer block (with a causal mask), and print the attention output
//! shape plus a few values.
//!
//! Run with: `cargo run --example attention_demo`

use mlrust_core::Tensor;
use mlrust_transformers::utils::causal_mask;
use mlrust_transformers::{Embedding, TransformerBlock};

fn main() {
    let vocab = 50;
    let d_model = 32;
    let d_ff = 128;

    let emb = Embedding::new(vocab, d_model);
    let block = TransformerBlock::new(d_model, d_ff);

    // Pretend "tokens" — any indices in [0, vocab).
    let tokens = vec![3, 17, 8, 0, 41, 22];
    let x: Tensor = emb.forward(&tokens);
    println!("embedded shape: {:?}", x.shape());

    let mask = causal_mask(tokens.len());
    let y = block.forward_with_mask(&x, Some(&mask));

    println!("output shape:   {:?}", y.shape());
    let row0 = y.data().row(0).to_owned();
    println!(
        "output row 0:   [{:.3}, {:.3}, {:.3}, ...]",
        row0[0], row0[1], row0[2]
    );

    let max_abs = y.data().iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    println!("max |y|:        {max_abs:.3}");
}
