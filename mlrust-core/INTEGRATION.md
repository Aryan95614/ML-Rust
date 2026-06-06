# Integration Guide: Using mlrust-core

This document explains how `mlrust` and `MLRustFoundation` can integrate with `mlrust-core`.

## Overview

```
┌─────────────────────────────────────────────────┐
│              mlrust-core                        │
│  (Tensor, Autograd, Basic Ops, Traits)         │
└─────────────────┬───────────────────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
┌───────▼────────┐  ┌────────▼───────────┐
│    mlrust      │  │ MLRustFoundation   │
│  (Transformers)│  │      (CNNs)        │
└────────────────┘  └────────────────────┘
```

## For `mlrust` Integration

### 1. Update Cargo.toml

```toml
[dependencies]
mlrust-core = { path = "../mlrust-core" }
# Keep existing dependencies
ndarray = "0.17"
smallvec = "1.11"
rand = "0.9.2"
```

### 2. Replace Internal Tensor with Core Tensor

**Before:**
```rust
// mlrust/src/tensor/data.rs
pub struct Tensor {
    inner: Rc<RefCell<TensorData>>,
}
```

**After:**
```rust
// mlrust/src/lib.rs
pub use mlrust_core::{Tensor, Layer, Optimizer};

// Your existing code now imports Tensor from core
```

### 3. Extend Core Operations

Keep your advanced operations (softmax, log, clamp, masked_fill) as extensions:

```rust
// mlrust/src/tensor_ext.rs
use mlrust_core::Tensor;

pub trait TensorExt {
    fn softmax(&self) -> Tensor;
    fn log(&self) -> Tensor;
    fn clamp(&self, min: f32, max: f32) -> Tensor;
    fn masked_fill(&self, mask: &Tensor, value: f32) -> Tensor;
}

impl TensorExt for Tensor {
    fn softmax(&self) -> Tensor {
        // Your existing softmax implementation
    }
    // ... other methods
}
```

### 4. Update Layer Trait

**Before:**
```rust
pub trait Layer {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn params(&mut self) -> ParamVec;  // Using smallvec
}
```

**After:**
```rust
use mlrust_core::{Layer, Tensor};
use smallvec::SmallVec;

// Keep your ParamVec alias
pub type ParamVec<'a> = SmallVec<[&'a mut Tensor; 8]>;

// Implement wrapper trait that converts to/from ParamVec
pub trait MLRustLayer: Layer {
    fn params_vec(&mut self) -> ParamVec;
}
```

### 5. Layers Stay Mostly the Same

```rust
// mlrust/src/layers/dense.rs
use mlrust_core::{Tensor, Layer};

pub struct Dense {
    weights: Tensor,
    bias: Tensor,
    // ... rest of your implementation
}

impl Layer for Dense {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.matmul(&self.weights).add(&self.bias)
    }

    fn params(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights, &mut self.bias]
    }
}

// Your >> operator syntax still works!
```

### 6. Keep Your Advanced Features

```rust
// mlrust/src/layers/attention.rs - NO CHANGES NEEDED
// mlrust/src/layers/layernorm.rs - NO CHANGES NEEDED
// mlrust/src/optim/adam.rs - Implement core::Optimizer trait
// mlrust/src/optim/scheduler.rs - NO CHANGES NEEDED
```

## For `MLRustFoundation` Integration

### 1. Update Cargo.toml

```toml
[dependencies]
mlrust-core = { path = "../mlrust-core" }
ndarray = "0.15"
# ... rest of dependencies
```

### 2. Replace Tensor Type

**Before:**
```rust
pub struct Tensor {
    data: Arc<Mutex<Array2<f32>>>,
    // ... complex error handling
}
```

**After:**
```rust
pub use mlrust_core::Tensor;

// Wrap it if you need Result types
pub type MLResult<T> = Result<T, MLError>;

pub struct SafeTensor {
    inner: Tensor,
}

impl SafeTensor {
    pub fn forward(&self, input: &Tensor) -> MLResult<Tensor> {
        // Add error checking wrapper
        Ok(self.inner.forward(input))
    }
}
```

### 3. Update Layer Implementations

**Before:**
```rust
impl Layer for Conv2D {
    fn forward(&self, input: &Tensor) -> Result<Tensor, MLError> {
        // ...
    }
}
```

**After:**
```rust
use mlrust_core::{Layer, Tensor};

// Option 1: Use core Layer directly
impl Layer for Conv2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Remove Result, add panics or unwraps
    }

    fn params(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights, &mut self.bias]
    }
}

// Option 2: Keep your Result-based trait
pub trait MLFoundationLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor, MLError>;
}

// And implement both
```

### 4. Optimizer Integration

```rust
use mlrust_core::Optimizer;

pub struct SGD {
    lr: f32,
    momentum: Option<f32>,
}

impl Optimizer for SGD {
    fn step(&self, params: &mut [&mut Tensor]) {
        for param in params {
            if let Some(grad) = param.grad() {
                let mut data = param.data_mut();
                *data = &*data - &(grad * self.lr);
            }
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}
```

## Shared Concepts

### Both Libraries Can Now Share:

1. **Tensor Type** - Same underlying representation
2. **Autograd** - Same computation graph mechanism
3. **Basic Operations** - add, mul, matmul, relu, sigmoid
4. **Traits** - Layer, Optimizer, Loss interfaces

### What Each Library Keeps:

**mlrust:**
- Transformer layers (Attention, Embedding, PositionalEncoding)
- Advanced operations (softmax, log, clamp, masked_fill)
- Learning rate schedulers
- Adam optimizer
- Training utilities and progress bars

**MLRustFoundation:**
- CNN layers (Conv2D, MaxPool2D, Flatten)
- Result-based error handling
- Model serialization
- MNIST data loading
- CLI tools

## Migration Strategy

### Phase 1: Parallel Development
- Keep existing code working
- Add mlrust-core as dependency
- Create adapter layer

### Phase 2: Gradual Migration
- Migrate basic operations first
- Update tests to use core types
- Keep advanced features separate

### Phase 3: Full Integration
- Replace internal Tensor with core Tensor
- Remove duplicate code
- Optimize for core integration

## Example: Full Integration

```rust
// mlrust/src/lib.rs
pub use mlrust_core::{Tensor, Layer, Optimizer};

// Re-export core with extensions
pub mod tensor {
    pub use mlrust_core::Tensor;

    // Add your extensions
    pub trait TensorExt {
        fn softmax(&self) -> Tensor;
        fn log(&self) -> Tensor;
    }

    impl TensorExt for Tensor {
        // Your implementations
    }
}

// Your advanced layers
pub mod layers {
    pub use mlrust_core::Layer;

    mod attention;
    mod embedding;
    mod layernorm;

    pub use attention::*;
    pub use embedding::*;
    pub use layernorm::*;
}

// Your optimizers
pub mod optim {
    pub use mlrust_core::Optimizer;

    mod adam;
    mod scheduler;

    pub use adam::*;
    pub use scheduler::*;
}
```

## Benefits of Integration

### Code Reuse
- Share 2000+ lines of core tensor/autograd code
- No duplication of basic operations
- Consistent behavior across libraries

### Interoperability
- Models from mlrust can use layers from MLRustFoundation
- Mix CNN and Transformer layers
- Shared optimizer implementations

### Maintenance
- Bug fixes in core benefit both libraries
- Single source of truth for autograd
- Easier to maintain and test

### Performance
- Optimizations in core benefit everyone
- Can add SIMD/GPU support to core
- Reduced compile times (shared dependencies)

## Compatibility Matrix

| Feature | mlrust-core | mlrust | MLRustFoundation |
|---------|-------------|--------|------------------|
| Tensor | ✅ | Uses core | Uses core |
| Autograd | ✅ | Uses core | Uses core |
| Add/Mul/MatMul | ✅ | Uses core | Uses core |
| ReLU/Sigmoid | ✅ | Uses core | Uses core |
| Softmax | ❌ | ✅ | Can use mlrust's |
| LayerNorm | ❌ | ✅ | Can use mlrust's |
| Attention | ❌ | ✅ | Can use mlrust's |
| Conv2D | ❌ | ❌ | ✅ |
| MaxPool2D | ❌ | ❌ | ✅ |
| Adam | ❌ | ✅ | Can use mlrust's |
| SGD | ❌ | ✅ | ✅ Both have it |
| Schedulers | ❌ | ✅ | Can use mlrust's |
| Error Handling | Basic | Panics | Result types |
| Serialization | ❌ | ❌ | ✅ |

---

**Next Steps:**
1. Review this integration guide
2. Create adapter layers for smooth migration
3. Update tests to use core types
4. Gradually migrate existing code
5. Remove duplicate implementations
