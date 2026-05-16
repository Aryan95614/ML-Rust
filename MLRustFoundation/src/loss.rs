//! Loss functions for neural network training
//!
//! Provides differentiable loss functions used to optimize neural networks.

use ndarray::{ArrayD, Axis};
use crate::tensor::data::Tensor;
use crate::tensor::autograd::GradFn;

/// Computes log-softmax for numerical stability
///
/// Uses the log-sum-exp trick: log(softmax(x)) = x - log(Σ exp(x))
/// by subtracting the max value before exponentiation to avoid overflow.
fn log_softmax(input: &Tensor) -> Tensor {
    let input_data = input.get_data();
    let mut output_data = ArrayD::zeros(input_data.shape());

    let axis = input_data.ndim() - 1;

    input_data.axis_iter(Axis(axis)).zip(output_data.axis_iter_mut(Axis(axis)))
        .for_each(|(input_slice, mut output_slice)| {
            let max_val = input_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_values: Vec<f32> = input_slice.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_values.iter().sum();
            let log_sum_exp = sum_exp.ln();

            for (i, &exp_val) in exp_values.iter().enumerate() {
                output_slice[i] = (exp_val.ln() - log_sum_exp);
            }
        });

    Tensor::new(output_data, input.shape.clone(), input.requires_grad)
}

/// Computes the cross-entropy loss for multi-class classification
///
/// # Arguments
///
/// * `logits` - Unnormalized model outputs (shape: batch_size × num_classes)
/// * `targets` - Ground truth class indices (shape: batch_size)
///
/// # Returns
///
/// A scalar tensor containing the mean cross-entropy loss across the batch
///
/// # Panics
///
/// Panics if:
/// - Logits are not 2D
/// - Targets are not 1D
/// - Batch sizes don't match
///
/// # Automatic Differentiation
///
/// Efficiently computes the gradient:
/// ∂L/∂logits[i,j] = (softmax[i,j] - 1) / batch_size if j == target[i]
///                   softmax[i,j] / batch_size otherwise
///
/// # Examples
///
/// ```no_run
/// use mlrust::loss::cross_entropy_loss;
/// use mlrust::tensor::Tensor;
///
/// let logits = Tensor::zeros(&[32, 10], true);  // 32 samples, 10 classes
/// let targets = Tensor::zeros(&[32], false);     // class indices
/// let loss = cross_entropy_loss(&logits, &targets);
/// ```
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    assert_eq!(logits.ndim(), 2, "Logits must be 2D (batch_size, num_classes)");
    assert_eq!(targets.ndim(), 1, "Targets must be 1D (batch_size)");
    assert_eq!(logits.shape.0[0], targets.shape.0[0], "Batch sizes must match");

    let log_probs = log_softmax(logits);
    let log_probs_data = log_probs.get_data();
    let targets_data = targets.get_data();

    let batch_size = logits.shape.0[0];
    let mut loss_sum = 0.0;

    for i in 0..batch_size {
        let target_class = targets_data[[i]].round() as usize;
        loss_sum -= log_probs_data[[i, target_class]];
    }

    let loss_data = ArrayD::from_elem(ndarray::IxDyn(&[]), loss_sum / batch_size as f32);
    let requires_grad = logits.requires_grad;

    let mut loss = Tensor::new(loss_data, vec![], requires_grad);

    if requires_grad {
        loss.grad_fn = Some(Box::new(GradFn::CrossEntropyLoss {
            logits: logits.clone(),
            targets: targets.clone(),
        }));
    }

    loss
}

