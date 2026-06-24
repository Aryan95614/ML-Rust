# Introduction

Welcome to the official documentation for MLRust, a lightweight deep learning library built from scratch in Rust. This documentation is designed to be a progressive and comprehensive guide to understanding and using the library, whether you are a beginner learning the fundamentals of machine learning or an experienced developer exploring how to implement these concepts in Rust.

## What is MLRust?

MLRust is an educational deep learning framework that demonstrates core neural network concepts using pure Rust. It is built with zero ML dependencies and features a PyTorch-like API. The library is designed to be simple, readable, and easy to understand, making it an excellent tool for learning.

## Philosophy

The core philosophy behind MLRust is **clarity over performance**. While the library is written in Rust and is reasonably fast for small-scale experiments, it is not intended for production use. Instead, every component is designed to be as transparent and straightforward as possible, helping you understand the inner workings of a deep learning library.

## Key Features

- **Tensor Operations**: A custom `Tensor` struct with support for automatic differentiation (autograd).
- **Neural Network Layers**: A suite of common layers, including `Dense`, `ReLU`, `Sigmoid`, and `LayerNorm`.
- **Intuitive Model Building**: A unique `>>` operator syntax for chaining layers into a `Sequential` model.
- **Optimizers**: Standard optimizers like `SGD` to update model parameters.
- **Loss Functions**: Common loss functions such as `MSE` and `BCE` for training.

## How to Use This Documentation

This documentation is structured to be read progressively. We recommend starting with the core concepts and moving through the sections in order:

1.  **The Tensor**: Learn about the fundamental data structure of the library.
2.  **Automatic Differentiation**: Understand how gradients are computed automatically.
3.  **Building Models**: Discover how to create neural network models.
4.  **Layers**: Dive into the details of each neural network layer.
5.  **Loss Functions**: Learn about the functions used to measure model performance.
6.  **Optimizers**: Understand how models are updated during training.
7.  **Training Loop**: See how all the components come together in a training loop.

Ready to get started? Let's dive into the first core concept: [The Tensor](tensor.md).