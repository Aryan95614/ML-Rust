//! # mlrust-transformers
//!
//! Transformer building blocks built on top of [`mlrust_core`].
//!
//! Forward-pass focused: attention, layernorm, feed-forward, and a stacked
//! encoder block. Operates on 2D tensors `(seq_len, d_model)` to stay
//! compatible with `mlrust-core::Tensor`.
//!
//! Backward passes go through the autograd ops that `mlrust-core` already
//! supports (matmul, add, mul, transpose, relu). Softmax / layernorm are
//! implemented as forward-only `ndarray` helpers for now — gradient hookup
//! lands in a later milestone.

#![allow(clippy::needless_range_loop)]

pub mod attention;
pub mod block;
pub mod embedding;
pub mod feedforward;
pub mod layernorm;
pub mod linear;
pub mod self_attention;
pub mod utils;

pub use attention::scaled_dot_product_attention;
pub use block::TransformerBlock;
pub use embedding::Embedding;
pub use feedforward::FeedForward;
pub use layernorm::LayerNorm;
pub use linear::Linear;
pub use self_attention::SelfAttention;
