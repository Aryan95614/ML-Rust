pub mod tensor;
pub mod ops;
pub mod layers;
pub mod optim;
pub mod loss;
pub mod train;
pub mod data;

/// Prelude module for convenient imports
///
/// Import everything you need with a single line:
///
/// ```ignore
/// use mlrust::prelude::*;
/// ```
///
/// This brings all commonly used types and functions into scope:
/// - Tensor operations
/// - Layer types (Dense, ReLU, Sigmoid, etc.)
/// - Optimizer (SGD)
/// - Loss functions (mse_loss, bce_loss)
pub mod prelude {
    pub use crate::tensor::*;
    pub use crate::ops::*;
    pub use crate::layers::*;
    pub use crate::optim::*;
    pub use crate::loss::*;
    pub use crate::train::*;
    pub use crate::data::*;
}
