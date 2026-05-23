//! Benchmarks for MLRustFoundation neural network operations
//!
//! Run with: `cargo bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use mlrust_foundation::layers::*;
use mlrust_foundation::tensor::Tensor;
use mlrust_foundation::ops::conv2d::{Stride, Padding};
use rand::thread_rng;

/// Benchmark CNN forward pass with different input sizes
fn benchmark_cnn_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("cnn_forward");

    for size in [28, 56, 112].iter() {
        let mut rng = thread_rng();
        let model = Input::image(1, *size, *size)
            >> Conv2D::new(1, 16, (3, 3), Stride(1, 1), Padding(1, 1), &mut rng)
                .expect("Failed to create Conv2D")
            >> ReLU::new()
            >> MaxPool2D::new((2, 2));

        let input = Tensor::ones(&[1, 1, *size, *size][..], false);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| black_box(model.forward(&input)))
            }
        );
    }
    group.finish();
}

/// Benchmark dense network forward pass with different batch sizes
fn benchmark_dense_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_forward");

    for batch_size in [1, 32, 128].iter() {
        let mut rng = thread_rng();
        let model = Input::image(1, 28, 28)
            >> Flatten::new()
            >> Dense::new(784, 128, &mut rng).expect("Failed to create Dense")
            >> ReLU::new()
            >> Dense::new(128, 10, &mut rng).expect("Failed to create Dense")
            >> Softmax::new();

        let input = Tensor::ones(&[*batch_size, 1, 28, 28][..], false);

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| black_box(model.forward(&input)))
            }
        );
    }
    group.finish();
}

/// Benchmark matrix multiplication with different sizes
fn benchmark_matmul(c: &mut Criterion) {
    use mlrust_foundation::ops::matmul;

    let mut group = c.benchmark_group("matmul");

    for size in [32, 128, 512].iter() {
        let a = Tensor::ones(&[*size, *size][..], false);
        let b = Tensor::ones(&[*size, *size][..], false);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bench, _| {
                bench.iter(|| black_box(matmul(&a, &b)))
            }
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_cnn_forward, benchmark_dense_forward, benchmark_matmul);
criterion_main!(benches);

