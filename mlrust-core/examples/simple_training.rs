use mlrust_core::{Tensor, mse_loss};

fn main() {
    println!("🎯 Complete Training Example with mlrust-core\n");
    println!("{}", "=".repeat(60));

    // ========================================================================
    // Simple Linear Model: y = w * x + b
    // ========================================================================

    println!("\n📊 Training a Simple Linear Model");
    println!("{}", "-".repeat(60));

    // True parameters: y = 3x + 2
    let true_w = 3.0;
    let true_b = 2.0;

    println!("True function: y = {:.1}x + {:.1}", true_w, true_b);

    // Generate training data
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data: Vec<f32> = x_data.iter().map(|x| true_w * x + true_b).collect();

    let x_train = Tensor::from_vec(x_data.clone(), &[5, 1], false);
    let y_train = Tensor::from_vec(y_data.clone(), &[5, 1], false);

    println!("Training data:");
    for (x, y) in x_data.iter().zip(y_data.iter()) {
        println!("  x={:.1} -> y={:.1}", x, y);
    }

    // Initialize parameters
    let w = Tensor::from_vec(vec![0.0], &[1, 1], true);
    let b = Tensor::from_vec(vec![0.0], &[1, 1], true);

    println!("\nInitial parameters: w={:.4}, b={:.4}",
             w.data()[[0, 0]], b.data()[[0, 0]]);

    // Training loop
    let learning_rate = 0.01;
    let epochs = 100;

    println!("\n🏋️ Training for {} epochs (lr={:.2})...", epochs, learning_rate);
    println!("\nEpoch      Loss       W          B");
    println!("{}", "-".repeat(60));

    for epoch in 0..=epochs {
        // Forward pass: y_pred = x * w + b
        let pred = x_train.matmul(&w).add(&b);

        // Compute loss
        let mut loss = mse_loss(&pred, &y_train);

        // Backward pass
        loss.backward();

        // Get gradients
        let w_grad = w.grad().unwrap();
        let b_grad = b.grad().unwrap();

        // Update parameters (SGD step)
        let mut w_data = w.data_mut();
        *w_data = &*w_data - &(w_grad * learning_rate);
        drop(w_data);

        let mut b_data = b.data_mut();
        *b_data = &*b_data - &(b_grad * learning_rate);
        drop(b_data);

        // Zero gradients
        w.zero_grad();
        b.zero_grad();

        // Print progress
        if epoch % 10 == 0 {
            println!("{:5}    {:8.6}   {:8.4}   {:8.4}",
                     epoch,
                     loss.item(),
                     w.data()[[0, 0]],
                     b.data()[[0, 0]]);
        }
    }

    println!("{}", "=".repeat(60));
    println!("\n✅ Training Complete!\n");
    println!("Final Results:");
    println!("  Learned w = {:.4} (true: {:.1})", w.data()[[0, 0]], true_w);
    println!("  Learned b = {:.4} (true: {:.1})", b.data()[[0, 0]], true_b);
    println!("  Error w = {:.4}", (w.data()[[0, 0]] - true_w).abs());
    println!("  Error b = {:.4}", (b.data()[[0, 0]] - true_b).abs());

    // Test on new data
    println!("\n🧪 Testing on new data:");
    let test_x = vec![6.0, 7.0, 8.0];
    for x in test_x {
        let x_tensor = Tensor::from_vec(vec![x], &[1, 1], false);
        let pred = x_tensor.matmul(&w).add(&b);
        let true_y = true_w * x + true_b;
        let pred_y = pred.item();
        println!("  x={:.1} -> predicted={:.4}, true={:.1}, error={:.4}",
                 x, pred_y, true_y, (pred_y - true_y).abs());
    }

    println!("\n{}", "=".repeat(60));
    println!("✨ Success! mlrust-core is complete and working!");
}

