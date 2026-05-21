//! Simple example demonstrating automatic differentiation
//!
//! This example shows how MLRustFoundation tracks computational graphs
//! and computes gradients automatically using backpropagation.
//!
//! Run with: `cargo run --example simple_autograd`

use mlrust_foundation::tensor::Tensor;
use mlrust_foundation::ops::{add, mul};
use ndarray::{ArrayD, IxDyn};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MLRustFoundation Autograd Example ===\n");

    // Example 1: Simple gradient computation
    println!("Example 1: Computing gradients for y = a + b");
    println!("-------------------------------------------");

    let a = Tensor::ones(&[2, 2][..], true);  // requires_grad = true
    let b = Tensor::ones(&[2, 2][..], true);  // requires_grad = true

    println!("a = \n{:?}", a.get_data());
    println!("\nb = \n{:?}", b.get_data());

    let c = add(&a, &b);
    println!("\nc = a + b = \n{:?}", c.get_data());

    // Compute gradients
    c.backward()?;

    // Gradients of addition: ∂c/∂a = 1, ∂c/∂b = 1
    println!("\n∂c/∂a = \n{:?}", a.get_grad()?.unwrap());
    println!("\n∂c/∂b = \n{:?}", b.get_grad()?.unwrap());

    println!("\n{}\n", "=".repeat(50));

    // Example 2: Chain rule with multiplication
    println!("Example 2: Chain rule with y = (a * b) + a");
    println!("-------------------------------------------");

    let x = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2.0, 3.0, 4.0, 5.0])?,
        &[2, 2][..],
        true
    );
    let y = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0])?,
        &[2, 2][..],
        true
    );

    println!("x = \n{:?}", x.get_data());
    println!("\ny = \n{:?}", y.get_data());

    let z1 = mul(&x, &y);
    let result = add(&z1, &x);

    println!("\nresult = (x * y) + x = \n{:?}", result.get_data());

    // Compute gradients using chain rule
    result.backward()?;

    // ∂result/∂x = y + 1 (from both mul and add)
    // ∂result/∂y = x (from mul only)
    println!("\n∂result/∂x = y + 1 = \n{:?}", x.get_grad()?.unwrap());
    println!("\n∂result/∂y = x = \n{:?}", y.get_grad()?.unwrap());

    println!("\n{}\n", "=".repeat(50));

    // Example 3: Simple neural network gradient
    println!("Example 3: Simple computation graph");
    println!("-----------------------------------");

    let w = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, -0.5, 1.0, -1.0])?,
        &[2, 2][..],
        true
    );
    let input = Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0])?,
        &[2, 2][..],
        true
    );

    println!("weights = \n{:?}", w.get_data());
    println!("\ninput = \n{:?}", input.get_data());

    let output = mul(&w, &input);
    println!("\noutput = weights * input = \n{:?}", output.get_data());

    output.backward()?;

    println!("\n∂output/∂weights = \n{:?}", w.get_grad()?.unwrap());
    println!("\n∂output/∂input = \n{:?}", input.get_grad()?.unwrap());

    println!("\n{}", "=".repeat(50));
    println!("\nKey Takeaways:");
    println!("- Tensors with requires_grad=true track computational history");
    println!("- .backward() computes gradients automatically using chain rule");
    println!("- Gradients accumulate in each tensor's .grad field");
    println!("- This is the foundation for training neural networks!");

    Ok(())
}

