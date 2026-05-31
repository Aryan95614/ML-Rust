//! Core tensor implementation with automatic differentiation support

use ndarray::Array2;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use crate::autograd::{GradFn, GradientManager};
use crate::Shape;

/// Internal tensor data shared across clones
struct TensorData {
    data: Array2<f32>,
    grad_manager: GradientManager,
}

/// Core tensor type with automatic differentiation
///
/// Uses Rc<RefCell<>> for shared ownership and interior mutability.
/// Cloning a tensor creates a new reference to the same underlying data.
#[derive(Clone)]
pub struct Tensor {
    inner: Rc<RefCell<TensorData>>,
}

impl Tensor {
    /// Creates a new tensor from raw data
    pub fn new(data: Array2<f32>, requires_grad: bool) -> Self {
        Self {
            inner: Rc::new(RefCell::new(TensorData {
                data,
                grad_manager: GradientManager::new(requires_grad),
            })),
        }
    }

    /// Creates a tensor from a flat vector and shape
    pub fn from_vec(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Self {
        assert_eq!(shape.len(), 2, "Shape must be 2D");
        let array = Array2::from_shape_vec((shape[0], shape[1]), data)
            .expect("Shape and data length mismatch");
        Self::new(array, requires_grad)
    }

    /// Creates a tensor filled with zeros
    pub fn zeros(shape: Shape, requires_grad: bool) -> Self {
        Self::new(Array2::zeros(shape), requires_grad)
    }

    /// Creates a tensor filled with ones
    pub fn ones(shape: Shape, requires_grad: bool) -> Self {
        Self::new(Array2::ones(shape), requires_grad)
    }

    /// Creates a tensor with values from standard normal distribution
    pub fn randn(shape: Shape, requires_grad: bool) -> Self {
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..shape.0 * shape.1)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        Self::from_vec(data, &[shape.0, shape.1], requires_grad)
    }

    /// Returns the shape of the tensor
    pub fn shape(&self) -> Shape {
        let inner = self.inner.borrow();
        let s = inner.data.shape();
        (s[0], s[1])
    }

    /// Returns a clone of the tensor's data
    pub fn data(&self) -> Array2<f32> {
        self.inner.borrow().data.clone()
    }

    /// Returns a mutable reference to the tensor's data
    pub fn data_mut(&self) -> std::cell::RefMut<'_, Array2<f32>> {
        std::cell::RefMut::map(self.inner.borrow_mut(), |inner| &mut inner.data)
    }

    /// Sets the tensor's data
    pub fn set_data(&self, data: Array2<f32>) {
        self.inner.borrow_mut().data = data;
    }

    /// Returns a clone of the gradient (if exists)
    pub fn grad(&self) -> Option<Array2<f32>> {
        self.inner.borrow().grad_manager.grad().cloned()
    }

    /// Checks if gradient tracking is enabled
    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().grad_manager.requires_grad()
    }

    /// Returns the gradient function
    pub fn grad_fn(&self) -> Option<Arc<dyn GradFn>> {
        self.inner.borrow().grad_manager.grad_fn().cloned()
    }

    /// Sets the gradient function
    pub fn set_grad_fn(&self, grad_fn: Arc<dyn GradFn>) {
        self.inner.borrow_mut().grad_manager.set_grad_fn(grad_fn);
    }
