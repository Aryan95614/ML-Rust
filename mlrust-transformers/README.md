# mlrust-transformers

Transformer building blocks on top of [`mlrust-core`](../mlrust-core).

Forward-pass-focused. Targets the 2D `(seq_len, d_model)` tensor shape that
`mlrust-core` exposes — multi-head and batched variants come once the core
gains 3D tensor support.

## Status

Single-head, forward-only. All listed pieces land and are exercised by tests:

- [x] `Linear` (He init, weight + bias as `Tensor`)
- [x] `Embedding` (row-gather over a learnable table)
- [x] `scaled_dot_product_attention` (with optional additive mask)
- [x] `LayerNorm` (per-row, learnable γ / β)
- [x] `FeedForward` (Linear → ReLU → Linear)
- [x] `SelfAttention` (Q/K/V/out projections around the kernel)
- [x] `TransformerBlock` (Pre-LN; residual around attention and FFN)

What does **not** flow grad yet: softmax, layernorm, embedding gather. The
matmul / add / relu paths inside `Linear` and `FeedForward` do propagate
gradients via `mlrust-core`'s autograd. Full backward pass for the rest is
queued behind axis-wise reductions and `exp` / `div` landing in core.

## Usage

```rust
use mlrust_transformers::{Embedding, TransformerBlock};
use mlrust_transformers::utils::causal_mask;

let emb = Embedding::new(/* vocab */ 50, /* d_model */ 32);
let block = TransformerBlock::new(/* d_model */ 32, /* d_ff */ 128);

let tokens = [3, 17, 8, 0, 41, 22];
let x = emb.forward(&tokens);
let mask = causal_mask(tokens.len());
let y = block.forward_with_mask(&x, Some(&mask));

assert_eq!(y.shape(), (6, 32));
```

Run the bundled demo:

```sh
cargo run --example attention_demo
```

## Layout

```
src/
├── lib.rs
├── linear.rs            # Dense layer (He init)
├── embedding.rs         # Token table + row gather
├── attention.rs         # scaled_dot_product_attention(q, k, v, mask)
├── self_attention.rs    # Q/K/V/out projections around the kernel
├── layernorm.rs         # Per-row LN with learnable γ / β
├── feedforward.rs       # Linear → ReLU → Linear
├── block.rs             # Pre-LN transformer block
└── utils.rs             # softmax_rows, causal_mask
```

## License

MIT.
