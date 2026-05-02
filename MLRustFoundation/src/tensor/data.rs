//! Tensor data structures and operations
//!
//! This module provides the core `Tensor` type which represents multi-dimensional
//! arrays with automatic differentiation support.

use std::sync::{Arc, Mutex};
use ndarray::{ArrayD, IxDyn};
use crate::tensor::autograd::GradFn;
use crate::error::{MLError, Result};
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use serde::de::{Visitor, SeqAccess};
use std::fmt;

/// Shape represents the dimensions of a tensor
///
/// Can be constructed from slices, vectors, or tuples of dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shape(pub Vec<usize>);

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape(dims.to_vec())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape(dims)
    }
}

impl From<(usize, usize)> for Shape {
    fn from((a, b): (usize, usize)) -> Self {
        Shape(vec![a, b])
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from((a, b, c, d): (usize, usize, usize, usize)) -> Self {
        Shape(vec![a, b, c, d])
    }
}

impl Shape {
    /// Convert shape to ndarray's dynamic dimension type
    pub fn to_ixdyn(&self) -> IxDyn {
        IxDyn(&self.0)
    }

    /// Returns the number of dimensions
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if the shape has no dimensions
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the total number of elements (product of all dimensions)
    pub fn size(&self) -> usize {
        self.0.iter().product()
    }
}

/// Device specifies where tensor computation occurs
///
/// Currently CPU-only, with GPU support planned for future releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    /// CPU computation device
    Cpu,
}

/// Multi-dimensional array with automatic differentiation support
///
/// `Tensor` is the fundamental building block for neural network operations.
/// It supports forward computation and backward gradient propagation.
///
/// # Examples
///
/// ```no_run
/// use mlrust::tensor::{Tensor, Shape};
/// use ndarray::ArrayD;
///
/// // Create a tensor from data
/// let data = ArrayD::zeros(vec![2, 3].into());
/// let tensor = Tensor::new(data, &[2, 3], true);
///
/// // Create zero/one tensors
/// let zeros = Tensor::zeros(&[3, 4], false);
/// let ones = Tensor::ones(&[2, 2], true);
/// ```
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Arc<Mutex<ArrayD<f32>>>,
    pub grad: Arc<Mutex<Option<ArrayD<f32>>>>,
    pub grad_fn: Option<Box<GradFn>>,
    pub shape: Shape,
    pub device: Device,
    pub requires_grad: bool,
}

// Helper struct for serializing/deserializing ArrayD
#[derive(Serialize, Deserialize)]
struct SerializableArrayD {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let data_guard = self.data.lock()
            .map_err(|e| serde::ser::Error::custom(format!("Lock error: {}", e)))?;
        let serializable_data = SerializableArrayD {
            data: data_guard.iter().copied().collect(),
            shape: data_guard.shape().to_vec(),
        };

        (serializable_data, &self.shape, &self.device, &self.requires_grad).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TensorVisitor;

        impl<'de> Visitor<'de> for TensorVisitor {
            type Value = Tensor;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Tensor")
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let serializable_data: SerializableArrayD = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let shape: Shape = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let device: Device = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                let requires_grad: bool = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(3, &self))?;

                let data = ArrayD::from_shape_vec(IxDyn(&serializable_data.shape), serializable_data.data)
                    .map_err(serde::de::Error::custom)?;

                Ok(Tensor {
                    data: Arc::new(Mutex::new(data)),
                    grad: Arc::new(Mutex::new(None)),
                    grad_fn: None,
                    shape,
                    device,
                    requires_grad,
                })
            }
        }

        deserializer.deserialize_seq(TensorVisitor)
    }
}


impl Tensor {
    /// Creates a new tensor from data and shape
    ///
    /// # Arguments
    ///
    /// * `data` - The underlying array data
    /// * `shape` - The shape of the tensor
    /// * `requires_grad` - Whether to track gradients for this tensor
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance
    ///
    /// # Panics
    ///
    /// Panics if data length doesn't match the shape size
    pub fn new<S>(data: ArrayD<f32>, shape: S, requires_grad: bool) -> Self
    where
        S: Into<Shape>,
    {
        let shape = shape.into();
        assert_eq!(data.len(), shape.size(), "Data length and shape size mismatch");

        Tensor {
            data: Arc::new(Mutex::new(data)),
            grad: Arc::new(Mutex::new(None)),
            grad_fn: None,
            shape,
            device: Device::Cpu,
            requires_grad,
        }
    }

    /// Creates a tensor filled with zeros
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor
    /// * `requires_grad` - Whether to track gradients
    pub fn zeros<S>(shape: S, requires_grad: bool) -> Self
    where
        S: Into<Shape>,
    {
        let shape = shape.into();
        let data = ArrayD::zeros(shape.to_ixdyn());
        Tensor::new(data, shape, requires_grad)
    }

    /// Creates a tensor filled with ones
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor
    /// * `requires_grad` - Whether to track gradients
    pub fn ones<S>(shape: S, requires_grad: bool) -> Self
    where
        S: Into<Shape>,
    {
        let shape = shape.into();
        let data = ArrayD::ones(shape.to_ixdyn());
        Tensor::new(data, shape, requires_grad)
    }

    /// Computes gradients by backpropagation
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if gradients are not tracked
    ///
    /// # Errors
    ///
    /// Returns `MLError::GradientError` if `requires_grad` is false
    pub fn backward(&self) -> Result<()> {
        if !self.requires_grad {
            return Err(MLError::GradientError(
                "Cannot compute gradients: requires_grad is false".to_string()
            ));
        }
        crate::tensor::autograd::backward(self)?;
        Ok(())
    }

    /// Sets the gradient for this tensor
    ///
    /// # Errors
    ///
    /// Returns `MLError::LockError` if the gradient lock cannot be acquired
    pub fn set_grad(&self, grad: ArrayD<f32>) -> Result<()> {
        *self.grad.lock()? = Some(grad);
        Ok(())
    }

    /// Gets a copy of the gradient
    ///
    /// # Errors
    ///
    /// Returns `MLError::LockError` if the gradient lock cannot be acquired
    pub fn get_grad(&self) -> Result<Option<ArrayD<f32>>> {
        Ok(self.grad.lock()?.clone())
    }

    /// Gets a copy of the data
    ///
    /// For internal use, this method currently uses unwrap for lock acquisition.
    /// Future versions will use proper error handling throughout.
    pub fn get_data(&self) -> ArrayD<f32> {
        self.data.lock().expect("Failed to acquire data lock").clone()
    }

    /// Tries to get a copy of the data with proper error handling
    ///
    /// # Errors
    ///
    /// Returns `MLError::LockError` if the data lock cannot be acquired
    pub fn try_get_data(&self) -> Result<ArrayD<f32>> {
        Ok(self.data.lock()?.clone())
    }

    /// Gets a mutable reference to the data
    ///
    /// # Errors
    ///
    /// Returns `MLError::LockError` if the data lock cannot be acquired
    pub fn get_data_mut(&self) -> Result<std::sync::MutexGuard<'_, ArrayD<f32>>> {
        Ok(self.data.lock()?)
    }

    /// Returns the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
}
