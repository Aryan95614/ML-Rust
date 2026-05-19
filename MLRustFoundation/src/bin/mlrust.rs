use clap::{Parser, Subcommand};
use mlrust_foundation::layers::input::Input;
use mlrust_foundation::layers::conv2d::Conv2D;
use mlrust_foundation::ops::conv2d::{Stride, Padding};  // Import from ops, not layers
use mlrust_foundation::layers::activations::{ReLU, Softmax};
use mlrust_foundation::layers::maxpool::MaxPool2D;
use mlrust_foundation::layers::flatten::Flatten;
use mlrust_foundation::layers::dense::Dense;
use mlrust_foundation::optim::sgd::SGD;
use mlrust_foundation::data::mnist;
use mlrust_foundation::train::Trainer;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Train a model
    Train(TrainArgs),
    /// Infer with a trained model
    Infer(InferArgs),
    /// Run benchmarks
    Benchmark(BenchmarkArgs),
}

#[derive(Parser, Debug)]
struct TrainArgs {
    /// Dataset to use (e.g., mnist)
    #[arg(long)]
    dataset: String,
    /// Model architecture to use (e.g., cnn, dense)
    #[arg(long)]
    model: String,
    /// Number of training epochs
    #[arg(long, default_value_t = 10)]
    epochs: usize,
    /// Learning rate
    #[arg(long, default_value_t = 0.01)]
    lr: f32,
    /// Batch size
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    /// Random seed for reproducibility
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Parser, Debug)]
struct InferArgs {
    /// Path to the trained model file
    #[arg(long)]
    model: String,
    /// Path to the image file for inference
    #[arg(long)]
    image: String,
}

#[derive(Parser, Debug)]
struct BenchmarkArgs {
    /// Kind of benchmark to run (e.g., cnn_forward)
    #[arg(long)]
    kind: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Train(args) => {
            println!("Training model: {} on dataset: {}", args.model, args.dataset);
            println!("Epochs: {}, LR: {}, Batch Size: {}, Seed: {}", args.epochs, args.lr, args.batch_size, args.seed);

            // Set random seed
            let mut rng = StdRng::seed_from_u64(args.seed);

            // Load MNIST data
            let (train_images, train_labels, test_images, test_labels) = mnist::load_mnist()?;

            // Build model
            let model: Box<dyn mlrust_foundation::layers::layer::Layer> = match args.model.as_str() {
                "cnn" => {
                    Box::new(
                        Input::image(1, 28, 28)
                            >> Conv2D::new(1, 32, (3, 3), Stride(1, 1), Padding(1, 1), &mut rng)
                            >> ReLU::new()
                            >> MaxPool2D::new((2, 2))
                            >> Conv2D::new(32, 64, (3, 3), Stride(1, 1), Padding(1, 1), &mut rng)
                            >> ReLU::new()
                            >> MaxPool2D::new((2, 2))
                            >> Flatten::new()
                            >> Dense::new(64 * 7 * 7, 128, &mut rng)
                            >> ReLU::new()
                            >> Dense::new(128, 10, &mut rng)
                            >> Softmax::new(),
                    )
                },
                "dense" => {
                    Box::new(
                        Input::new(vec![1, 28, 28]) // Input for flattened image
                            >> Flatten::new()
                            >> Dense::new(28 * 28, 128, &mut rng)
                            >> ReLU::new()
                            >> Dense::new(128, 10, &mut rng)
                            >> Softmax::new(),
                    )
                },
                _ => panic!("Unknown model architecture: {}", args.model),
            };

            // Optimizer
            let optimizer = SGD::new(args.lr, None); // No momentum for now

            // Trainer
            let mut trainer = Trainer::new(model, optimizer, args.epochs, args.batch_size);
            trainer.fit(train_images, train_labels, test_images, test_labels, &mut rng);
        },
        Commands::Infer(args) => {
            println!("Inferring with model: {} on image: {}", args.model, args.image);
            // TODO: Implement inference logic
        },
        Commands::Benchmark(args) => {
            println!("Running benchmark: {}", args.kind);
            // TODO: Implement benchmark logic
        },
    }

    Ok(())
}

