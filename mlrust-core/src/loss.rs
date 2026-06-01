//! Loss functions

use crate::Tensor;
use ndarray::Array2;

/// Mean Squared Error loss
///
/// Computes: mean((pred - target)^2)
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(
        pred.shape(),
        target.shape(),
        "MSE loss: pred and target must have the same shape"
    );

    // Compute difference: pred - target
    let neg_target_data = target.data().mapv(|x| -x);
    let neg_target = Tensor::new(neg_target_data, false);
    let diff = pred.add(&neg_target);

    // Square the differences
    let squared = diff.mul(&diff);

    squared.mean()
}

/// Binary Cross Entropy loss
///
/// Computes: -mean(target * log(pred) + (1 - target) * log(1 - pred))
///
/// Predictions should be in [0, 1] range (i.e., after sigmoid)
pub fn bce_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(
        pred.shape(),
        target.shape(),
        "BCE loss: pred and target must have the same shape"
    );

    const EPSILON: f32 = 1e-7;

    // Clamp predictions to [epsilon, 1 - epsilon] for numerical stability
    let pred_stable = pred.clamp(EPSILON, 1.0 - EPSILON);

    // Compute log(pred)
    let log_pred = pred_stable.log();

    // Compute 1 - pred
    let neg_one = Tensor::new(Array2::from_elem(pred.shape(), -1.0), false);
    let one_minus_pred = pred_stable.mul(&neg_one).add(&Tensor::ones(pred.shape(), false));

    // Compute log(1 - pred)
    let log_one_minus_pred = one_minus_pred.log();

    // Compute target * log(pred)
    let term1 = target.mul(&log_pred);

    // Compute (1 - target)
    let neg_one_target = Tensor::new(Array2::from_elem(target.shape(), -1.0), false);
    let one_minus_target = target.mul(&neg_one_target).add(&Tensor::ones(target.shape(), false));

    // Compute (1 - target) * log(1 - pred)
    let term2 = one_minus_target.mul(&log_one_minus_pred);

    // Sum the terms: target * log(pred) + (1 - target) * log(1 - pred)
    let sum = term1.add(&term2);

    // Take the mean and negate
    sum.mean().neg()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss_simple() {
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], true);
        let target = Tensor::from_vec(vec![2.0, 2.0, 2.0], &[1, 3], false);

        let loss = mse_loss(&pred, &target);

        assert_eq!(loss.shape(), (1, 1));
        let expected = 2.0 / 3.0; // diff = [-1, 0, 1], squared = [1, 0, 1], mean = 2/3
        assert!((loss.data()[[0, 0]] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_bce_loss_simple() {
        let pred = Tensor::from_vec(vec![0.5, 0.5], &[1, 2], true);
        let target = Tensor::from_vec(vec![1.0, 0.0], &[1, 2], false);

        let loss = bce_loss(&pred, &target);

        assert_eq!(loss.shape(), (1, 1));
        let expected = -(0.5_f32.ln());
        assert!((loss.data()[[0, 0]] - expected).abs() < 0.01);
    }

    #[test]
    fn test_mse_loss_backward() {
        let pred = Tensor::from_vec(vec![2.0, 3.0], &[1, 2], true);
        let target = Tensor::from_vec(vec![1.0, 2.0], &[1, 2], false);

        let mut loss = mse_loss(&pred, &target);
        loss.backward();

        assert!(pred.grad().is_some());
        assert!(loss.grad().is_some());
    }
}

