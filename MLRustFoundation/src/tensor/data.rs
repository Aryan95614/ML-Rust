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
