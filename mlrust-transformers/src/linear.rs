//! Dense (fully-connected) layer.
//!
//! `y = x @ W + b` where `W` is `(in_features, out_features)` and the bias is
//! broadcast across the sequence dimension.

use mlrust_core::{Layer, Tensor};
use ndarray::Array2;
use rand_distr::{Distribution, Normal};

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    /// He init: weights sampled from N(0, 2/fan_in). Better than Xavier for
    /// stacks that include ReLU (which kills half the activations and would
    /// otherwise shrink variance through depth).
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let std = (2.0_f32 / in_features as f32).sqrt();
        let normal = Normal::new(0.0, std).expect("valid normal");
        let mut rng = rand::rng();

        let w_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let weight_arr =
            Array2::from_shape_vec((in_features, out_features), w_data).expect("shape ok");
        let weight = Tensor::new(weight_arr, true);
        let bias = Tensor::new(Array2::zeros((1, out_features)), true);

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let projected = input.matmul(&self.weight);
        // Bias is (1, out_features); ndarray's broadcasting in mlrust-core's
        // `add` reduces gradient back to (1, out_features) automatically.
        projected.add(&self.bias)
    }

    fn params(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_shape_matches_out_features() {
        let layer = Linear::new(8, 4);
        let x = Tensor::randn((3, 8), false);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), (3, 4));
    }

    #[test]
    fn weight_init_has_expected_scale() {
        let layer = Linear::new(64, 32);
        let w = layer.weight.data();
        let std = (w.iter().map(|x| x * x).sum::<f32>() / w.len() as f32).sqrt();
        // He init: expected std ≈ sqrt(2/64) ≈ 0.177; allow 30% slack.
        assert!(std > 0.12 && std < 0.25, "got std={std}");
    }

    #[test]
    fn bias_starts_at_zero() {
        let layer = Linear::new(4, 4);
        assert!(layer.bias.data().iter().all(|x| *x == 0.0));
    }
}
