use mlrust_core::Tensor;

fn main() {
    println!("🦀 mlrust-core: Basic Usage Example\n");
    println!("{}",  "=".repeat(60));

    // ========================================================================
    // 1. Tensor Creation
    // ========================================================================
    println!("\n📦 1. Creating Tensors");
    println!("{}", "-".repeat(60));

    let a = Tensor::ones((2, 3), true);
    let b = Tensor::zeros((2, 3), false);
    let c = Tensor::randn((2, 3), true);

    println!("Created tensor a: {:?}", a);
    println!("Created tensor b: {:?}", b);
    println!("Created tensor c: {:?}", c);

    // ========================================================================
    // 2. Basic Operations
    // ========================================================================
    println!("\n➕ 2. Basic Operations");
    println!("{}", "-".repeat(60));

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let y = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], &[2, 2], true);

    let sum = x.add(&y);
    let prod = x.mul(&y);

    println!("x: {:?}", x.data());
    println!("y: {:?}", y.data());
    println!("x + y: {:?}", sum.data());
    println!("x * y: {:?}", prod.data());

    // ========================================================================
    // 3. Matrix Multiplication
    // ========================================================================
    println!("\n🔢 3. Matrix Multiplication");
    println!("{}", "-".repeat(60));

    let m1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let m2 = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], true);
    let result = m1.matmul(&m2);

    println!("m1: {:?}", m1.data());
    println!("m2: {:?}", m2.data());
    println!("m1 @ m2: {:?}", result.data());

    // ========================================================================
    // 4. Activation Functions
    // ========================================================================
    println!("\n✨ 4. Activation Functions");
    println!("{}", "-".repeat(60));

    let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[1, 5], true);
    let relu_out = input.relu();
    let sigmoid_out = input.sigmoid();

    println!("Input: {:?}", input.data());
    println!("ReLU: {:?}", relu_out.data());
    println!("Sigmoid: {:?}", sigmoid_out.data());

    // ========================================================================
    // 5. Automatic Differentiation
    // ========================================================================
    println!("\n🎯 5. Automatic Differentiation");
    println!("{}", "-".repeat(60));

    // Simple computation: loss = (a * b).sum()
    let a = Tensor::from_vec(vec![2.0, 3.0], &[1, 2], true);
    let b = Tensor::from_vec(vec![4.0, 5.0], &[1, 2], true);

    println!("a: {:?}", a.data());
    println!("b: {:?}", b.data());

    let c = a.mul(&b);
    println!("c = a * b: {:?}", c.data());

    let mut loss = c.sum();
    println!("loss = c.sum(): {:.4}", loss.item());

    // Backpropagate
    loss.backward();

    println!("\nGradients:");
    println!("∂loss/∂a: {:?}", a.grad().unwrap());
    println!("∂loss/∂b: {:?}", b.grad().unwrap());

    // ========================================================================
    // 6. Complex Computation Graph
    // ========================================================================
    println!("\n🌲 6. Complex Computation Graph");
    println!("{}", "-".repeat(60));

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let w = Tensor::from_vec(vec![0.5, 0.5, 0.5, 0.5], &[2, 2], true);
    let b = Tensor::from_vec(vec![1.0, 1.0], &[1, 2], true);

    // Forward: y = ReLU(x @ w + b)
    let linear = x.matmul(&w);
    let with_bias = linear.add(&b);
    let activated = with_bias.relu();
    let mut loss = activated.sum();

    println!("x: {:?}", x.data());
    println!("w: {:?}", w.data());
    println!("b: {:?}", b.data());
    println!("linear (x @ w): {:?}", linear.data());
    println!("with_bias (linear + b): {:?}", with_bias.data());
    println!("activated (ReLU): {:?}", activated.data());
    println!("loss: {:.4}", loss.item());

    // Backward
    loss.backward();

    println!("\nGradients:");
    println!("∂loss/∂x: {:?}", x.grad().unwrap());
    println!("∂loss/∂w: {:?}", w.grad().unwrap());
    println!("∂loss/∂b: {:?}", b.grad().unwrap());

    // ========================================================================
    // 7. Reduction Operations
    // ========================================================================
    println!("\n📊 7. Reduction Operations");
    println!("{}", "-".repeat(60));

    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let sum = tensor.sum();
    let mean = tensor.mean();

    println!("Tensor: {:?}", tensor.data());
    println!("Sum: {:.4}", sum.item());
    println!("Mean: {:.4}", mean.item());

    println!("\n{}",  "=".repeat(60));
    println!("✅ All examples completed successfully!");
}

