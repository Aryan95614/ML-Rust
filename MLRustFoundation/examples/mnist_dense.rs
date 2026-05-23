// Simple MLP for MNIST

use mlrust_foundation::data::mnist::load_mnist;
use mlrust_foundation::layers::input::Input;
use mlrust_foundation::layers::dense::Dense;
use mlrust_foundation::layers::flatten::Flatten;
use mlrust_foundation::layers::activations::{ReLU, Softmax};
use mlrust_foundation::optim::sgd::SGD;
use mlrust_foundation::train::Trainer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MNIST Dense Network\n");

    let (train_images, train_labels, test_images, test_labels) = load_mnist()?;
    let mut rng = rand::thread_rng();

    let model = Input::image(1, 28, 28)
        >> Flatten::new()
        >> Dense::new(784, 256, &mut rng)
        >> ReLU::new()
        >> Dense::new(256, 128, &mut rng)
        >> ReLU::new()
        >> Dense::new(128, 10, &mut rng)
        >> Softmax::new();

    let optimizer = SGD::new(0.1, Some(0.9));
    let mut trainer = Trainer::new(Box::new(model), optimizer, 10, 128);

    trainer.fit(train_images, train_labels, test_images, test_labels, &mut rng);

    Ok(())
}

