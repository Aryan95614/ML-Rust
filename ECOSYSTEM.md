# MLRust Ecosystem

The MLRust project is a family of machine learning libraries I've been writing in Rust, sharing a common foundation.

```
ML-Rust/
├── mlrust-core/          # Core tensor & autograd (~1,200 lines)
├── mlrust/               # Transformers & NLP (~10,000 lines)
└── MLRustFoundation/     # CNNs & vision (~2,600 lines)
```

## Architecture

```
                     MLRust Ecosystem
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │          mlrust-core                  │
        │                                       │
        │  * Tensor (Rc<RefCell<>>)             │
        │  * Autograd (GradFn trait)            │
        │  * 11 core operations                 │
        │  * MSE & BCE loss                     │
        │  * Layer, Optimizer, Loss traits      │
        │                                       │
        │  ~1,200 lines | 8 tests passing       │
        └───────────────┬───────────────────────┘
                        │
            ┌───────────┴──────────┐
            │                      │
   ┌────────▼─────────┐   ┌───────▼──────────┐
   │ mlrust           │   │  MLRustFoundation│
   │                  │   │                  │
   │  Transformers    │   │  CNNs            │
   │  * Attention     │   │  * Conv2D        │
   │  * Embedding     │   │  * MaxPool2D     │
   │  * LayerNorm     │   │  * Flatten       │
   │  * Positional    │   │                  │
   │  * Dropout       │   │  * Result<T,E>   │
   │                  │   │  * Serialize     │
   │  Training        │   │  * CLI tools     │
   │  * Adam, SGD     │   │  * MNIST data    │
   │  * 4 Schedulers  │   │                  │
   │  * Grad monitor  │   │  SGD + momentum  │
   │  * Model summary │   │  Cross Entropy   │
   │                  │   │                  │
   │  326 tests       │   │  ~2,600 lines    │
   │  ~10,000 lines   │   │                  │
   └──────────────────┘   └──────────────────┘
```

## Library Comparison

| Feature | mlrust-core | mlrust | MLRustFoundation |
|---------|-------------|--------|------------------|
| Purpose | Foundation | NLP/Transformers | Computer Vision |
| Size | 1,200 lines | 10,000 lines | 2,600 lines |
| Tests | 8 | 326 | TBD |
| License | MIT | MIT | MIT |

## Core Operations (mlrust-core)

All three libraries share these operations:

| Operation | Description | Gradient |
|-----------|-------------|----------|
| `add` | Element-wise addition | yes |
| `mul` | Element-wise multiplication | yes |
| `matmul` | Matrix multiplication | yes |
| `sum` | Reduce to scalar | yes |
| `mean` | Average all elements | yes |
| `relu` | ReLU activation | yes |
| `sigmoid` | Sigmoid activation | yes |
| `log` | Natural logarithm | yes |
| `neg` | Negation | yes |
| `clamp` | Clamp to range | yes |
| `transpose` | Matrix transpose | yes |

## Feature Coverage

| Operation | mlrust-core | mlrust | MLRustFoundation |
|-----------|-------------|--------|------------------|
| Tensor creation | yes | via core | via core |
| add/mul/matmul | yes | via core | via core |
| relu/sigmoid | yes | via core | via core |
| transpose | yes | via core | via core |
| sum/mean | yes | via core | via core |
| Autograd | yes | via core | via core |
| Broadcasting | yes | via core | via core |

### Advanced Operations

| Feature | mlrust-core | mlrust | MLRustFoundation |
|---------|-------------|--------|------------------|
| softmax | no | yes | no |
| log/exp | no | yes | no |
| clamp | no | yes | no |
| masked_fill | no | yes | no |
| mul_scalar | no | yes | no |

### Layers

| Layer Type | mlrust-core | mlrust | MLRustFoundation |
|------------|-------------|--------|------------------|
| Dense | no | yes | yes |
| Attention | no | yes | no |
| LayerNorm | no | yes | no |
| Embedding | no | yes | no |
| Positional Encoding | no | yes | no |
| Dropout | no | yes | no |
| Conv2D | no | no | yes |
| MaxPool2D | no | no | yes |
| Flatten | no | no | yes |
| Sequential | no | yes | no |

### Optimizers & Training

| Feature | mlrust-core | mlrust | MLRustFoundation |
|---------|-------------|--------|------------------|
| Optimizer trait | yes | via core | via core |
| SGD | no | yes | yes |
| Adam | no | yes | no |
| SGD + momentum | no | no | yes |
| StepLR | no | yes | no |
| ExponentialLR | no | yes | no |
| CosineAnnealingLR | no | yes | no |
| ReduceLROnPlateau | no | yes | no |
| Gradient monitoring | no | yes | no |
| Progress bars | no | yes | yes |
| Model summary | no | yes | no |

### Loss Functions

| Loss | mlrust-core | mlrust | MLRustFoundation |
|------|-------------|--------|------------------|
| Loss trait | yes | via core | via core |
| MSE | no | yes | no |
| BCE | no | yes | no |
| Cross Entropy | no | no | yes |

### Misc

| Feature | mlrust-core | mlrust | MLRustFoundation |
|---------|-------------|--------|------------------|
| Error handling | Basic (panic) | Panics | Result<T, E> |
| Serialization | no | no | yes (bincode) |
| CLI tools | no | no | yes |
| Data loading | no | no | yes (MNIST) |
| Model summary | no | yes | no |
| `>>` operator | no | yes | yes |

## Use Cases

### Use `mlrust-core` when
- Building your own ML library
- Learning about autograd internals
- You need minimal tensor operations
- You want maximum control and simplicity

### Use `mlrust` when
- Building transformers or NLP models
- You need attention mechanisms
- You want advanced training utilities
- You need learning rate schedulers
- Building language models or sequence tasks

### Use `MLRustFoundation` when
- Building CNNs for vision
- You need proper error handling (Result types)
- You want model serialization
- Training on MNIST or image datasets

## Integration Example

Both libraries can work together:

```rust
use mlrust_core::Tensor;
use mlrust::layers::{Attention, LayerNorm, Dense as MLRustDense};
use mlrust_foundation::layers::{Conv2D, MaxPool2D};

let model = Input::image(1, 28, 28)
    >> Conv2D::new(...)           // From MLRustFoundation
    >> MaxPool2D::new(...)         // From MLRustFoundation
    >> Flatten                     // From MLRustFoundation
    >> MLRustDense::new(...)       // From mlrust
    >> Attention::new(...)         // From mlrust
    >> LayerNorm::new(...);        // From mlrust
```

## Shared Concepts

All three libraries share:

1. Tensor type - Rc<RefCell<>> based for gradient sharing
2. Autograd mechanism - GradFn trait and backward passes
3. Layer trait - Common interface for forward/params
4. Optimizer trait - Common interface for parameter updates
5. Loss trait - Common interface for loss computation
6. `>>` operator - Model building syntax

## Dependencies

### mlrust-core
```toml
ndarray = "0.17"
rand = "0.9"
rand_distr = "0.5"
```

### mlrust
```toml
mlrust-core = { path = "../mlrust-core" }
ndarray = "0.17"
rand = "0.9.2"
rand_distr = "0.5"
smallvec = "1.11"
```

### MLRustFoundation
```toml
mlrust-core = { path = "../mlrust-core" }
ndarray = "0.15"
rand = "0.8"
clap = "4.5"
indicatif = "0.17"
serde = "1.0"
bincode = "1.3"
reqwest = "0.11"
```

## Performance Characteristics

| Library | Compile Time | Runtime | Memory Usage |
|---------|--------------|---------|--------------|
| mlrust-core | Fast (minimal deps) | Good | Low |
| mlrust | Medium | Good | Medium |
| MLRustFoundation | Slower (many deps) | Good | Medium |

## Future

### mlrust-core
- Consider 3D/4D tensor support
- Optional GPU backend trait
- f64 precision support

### mlrust
- Migrate to mlrust-core
- More transformer variants

### MLRustFoundation
- Migrate to mlrust-core
- Additional optimizers (Adam, RMSprop)
- Batch normalization

## License

All libraries: MIT.
