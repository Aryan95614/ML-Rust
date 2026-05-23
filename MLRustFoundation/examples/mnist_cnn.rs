// CNN example for MNIST
// Based on LeNet-5 architecture but smaller

use mlrust_foundation::data::mnist::load_mnist;
use mlrust_foundation::layers::input::Input;
use mlrust_foundation::layers::conv2d::Conv2D;
use mlrust_foundation::layers::maxpool::MaxPool2D;
use mlrust_foundation::layers::flatten::Flatten;
use mlrust_foundation::layers::dense::Dense;
use mlrust_foundation::layers::activations::{ReLU, Softmax};
use mlrust_foundation::ops::conv2d::{Stride, Padding};
use mlrust_foundation::optim::sgd::SGD;
use mlrust_foundation::train::Trainer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Training CNN on MNIST\n");

    println!("Loading data...");
    let (train_images, train_labels, test_images, test_labels) = load_mnist()?;
    println!("Train: {}, Test: {}", train_images.shape.0[0], test_images.shape.0[0]);

    // Build CNN model using operator chaining
    println!("Building CNN model...");
    let mut rng = rand::thread_rng();

    let model = Input::image(1, 28, 28)
        >> Conv2D::new(1, 16, (3, 3), Stride(1, 1), Padding(1, 1), &mut rng)
        >> ReLU::new()
        >> MaxPool2D::new((2, 2))
        >> Conv2D::new(16, 32, (3, 3), Stride(1, 1), Padding(1, 1), &mut rng)
        >> ReLU::new()
        >> MaxPool2D::new((2, 2))
        >> Flatten::new()
        >> Dense::new(32 * 7 * 7, 128, &mut rng)
        >> ReLU::new()
        >> Dense::new(128, 10, &mut rng)
        >> Softmax::new();

    // Setup training
    let optimizer = SGD::new(0.01, Some(0.9));
    let epochs = 5;
    let batch_size = 64;

    println!("Epochs: {}, Batch size: {}, LR: 0.01\n", epochs, batch_size);

    // Create trainer and start training
    let mut trainer = Trainer::new(
        Box::new(model),
        optimizer,
        epochs,
        batch_size,
    );

    trainer.fit(
        train_images,
        train_labels,
        test_images,
        test_labels,
        &mut rng,
    );

    Ok(())
}

