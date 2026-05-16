use crate::tensor::data::Tensor;
use smallvec::SmallVec;

// SGD with optional momentum
pub struct SGD {
    lr: f32,
    momentum: Option<f32>,
    velocity: SmallVec<[Option<Tensor>; 8]>,
}

impl SGD {
    pub fn new(lr: f32, momentum: Option<f32>) -> Self {
        SGD {
            lr,
            momentum,
            velocity: SmallVec::new(),
        }
    }

    pub fn step(&mut self, params: SmallVec<[&mut Tensor; 8]>) {
        if self.velocity.is_empty() {
            // Initialize velocity for the first step
            for param in params.iter() {
                self.velocity.push(None);
            }
        }

        for (i, param) in params.into_iter().enumerate() {
            if let Some(grad) = param.get_grad().expect("Failed to get gradient") {
                if let Some(momentum) = self.momentum {
                    // With momentum
                    let mut v = self.velocity[i].take().unwrap_or_else(|| {
                        Tensor::zeros(param.shape.clone(), false) // Initialize with zeros
                    });
                    {
                        let mut v_data = v.get_data_mut().expect("Failed to get velocity data");
                        *v_data = momentum * &*v_data + (1.0 - momentum) * &grad;

                        let mut param_data = param.get_data_mut().expect("Failed to get param data");
                        *param_data = &*param_data - self.lr * &*v_data;
                    }

                    self.velocity[i] = Some(v);
                } else {
                    // Without momentum
                    let mut param_data = param.get_data_mut().expect("Failed to get param data");
                    *param_data = &*param_data - self.lr * &grad;
                }
            }
        }
    }

    pub fn zero_grad(&mut self, params: SmallVec<[&mut Tensor; 8]>) {
        for param in params.into_iter() {
            *param.grad.lock().unwrap() = None;
        }
    }
}

