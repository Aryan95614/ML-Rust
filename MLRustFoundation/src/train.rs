use crate::layers::layer::Layer;
use crate::optim::sgd::SGD;
use crate::tensor::data::Tensor;
use crate::data::mnist::MnistBatcher;
use crate::loss::cross_entropy_loss;
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rand::Rng;
use ndarray;

pub struct Trainer {
    model: Box<dyn Layer>,
    optimizer: SGD,
    epochs: usize,
    batch_size: usize,
}

impl Trainer {
    pub fn new(model: Box<dyn Layer>, optimizer: SGD, epochs: usize, batch_size: usize) -> Self {
        Trainer {
            model,
            optimizer,
            epochs,
            batch_size,
        }
    }

    pub fn fit<R: Rng>(&mut self, train_images: Tensor, train_labels: Tensor, test_images: Tensor, test_labels: Tensor, rng: &mut R) {
        println!("Starting training...");

        let num_train_samples = train_images.shape.0[0];
        let num_test_samples = test_images.shape.0[0];

        for epoch in 1..=self.epochs {
            println!("\nEpoch {}/{}", epoch, self.epochs);

            // shuffle data
            let mut indices: Vec<usize> = (0..num_train_samples).collect();
            indices.shuffle(rng);

            // NOTE: this clones the entire dataset which is slow but works for MNIST
            // better solution would be to index directly during batching
            let shuffled_train_images_data = train_images.get_data().select(ndarray::Axis(0), &indices).to_owned();
            let shuffled_train_labels_data = train_labels.get_data().select(ndarray::Axis(0), &indices).to_owned();

            let shuffled_train_images = Tensor::new(shuffled_train_images_data, train_images.shape.0.clone(), false);
            let shuffled_train_labels = Tensor::new(shuffled_train_labels_data, train_labels.shape.0.clone(), false);


            let mut train_batcher = MnistBatcher::new(shuffled_train_images, shuffled_train_labels, self.batch_size);

            let num_batches = (num_train_samples + self.batch_size - 1) / self.batch_size;
            let pb = ProgressBar::new(num_batches as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg})")
                .unwrap()
                .progress_chars("#>-    "));

            let mut total_loss = 0.0;
            let mut correct_predictions = 0;
            let mut total_samples = 0;

            for (batch_idx, (images, labels)) in train_batcher.enumerate() {
                pb.set_message(format!("Batch {}/{}", batch_idx + 1, num_batches));

                // Forward pass
                let output = self.model.forward(&images);

                // Calculate loss
                let loss = cross_entropy_loss(&output, &labels);
                total_loss += loss.get_data()[[]];

                // Backward pass
                self.optimizer.zero_grad(self.model.params());
                loss.backward().expect("Failed to compute gradients");
                self.optimizer.step(self.model.params());

                // Calculate accuracy
                let predictions = output.get_data().map_axis(ndarray::Axis(1), |row| {
                    row.iter().enumerate().fold((0, f32::NEG_INFINITY), |(idx_max, val_max), (idx, &val)| {
                        if val > val_max { (idx, val) } else { (idx_max, val_max) }
                    }).0 as f32
                });
                let labels_data = labels.get_data();

                for i in 0..predictions.len() {
                    if predictions[[i]] == labels_data[[i]] {
                        correct_predictions += 1;
                    }
                    total_samples += 1;
                }
                pb.inc(1);
            }
            pb.finish_with_message("Training complete for epoch.");

            let avg_loss = total_loss / num_batches as f32;
            let accuracy = correct_predictions as f32 / total_samples as f32;
            println!("Train Loss: {:.4}, Train Accuracy: {:.4}", avg_loss, accuracy);

            // Evaluate on test set
            let mut test_batcher = MnistBatcher::new(test_images.clone(), test_labels.clone(), self.batch_size);
            let mut test_correct_predictions = 0;
            let mut test_total_samples = 0;

            for (images, labels) in test_batcher {
                let output = self.model.forward(&images);
                let predictions = output.get_data().map_axis(ndarray::Axis(1), |row| {
                    row.iter().enumerate().fold((0, f32::NEG_INFINITY), |(idx_max, val_max), (idx, &val)| {
                        if val > val_max { (idx, val) } else { (idx_max, val_max) }
                    }).0 as f32
                });
                let labels_data = labels.get_data();

                for i in 0..predictions.len() {
                    if predictions[[i]] == labels_data[[i]] {
                        test_correct_predictions += 1;
                    }
                    test_total_samples += 1;
                }
            }
            let test_accuracy = test_correct_predictions as f32 / test_total_samples as f32;
            println!("Test Accuracy: {:.4}", test_accuracy);
        }
    }
}
