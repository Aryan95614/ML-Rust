//! Model serialization and deserialization
//!
//! This module provides functionality for saving and loading trained models.

use serde::{Serialize, Deserialize};
use crate::tensor::data::Tensor;
use crate::layers::layer::Layer;
use crate::layers::dense::Dense;
use crate::layers::conv2d::Conv2D;
use crate::layers::activations::{ReLU, Softmax};
use crate::layers::maxpool::MaxPool2D;
use crate::layers::flatten::Flatten;
use crate::layers::input::Input;
use crate::ops::conv2d::{Stride, Padding};  // Import from ops, not layers
use crate::layers::builder::Sequential;
use std::fs::File;
use std::io::{BufWriter, BufReader};
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelParams {
    pub layers: Vec<LayerConfig>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum LayerConfig {
    Input(Input),
    Dense { input_size: usize, output_size: usize, weights: Tensor, bias: Tensor },
    Conv2D { input_channels: usize, output_channels: usize, kernel_size: (usize, usize), stride: Stride, padding: Padding, kernel: Tensor, bias: Tensor },
    ReLU,
    Softmax,
    MaxPool2D { kernel_size: (usize, usize) },
    Flatten,
}

/// Saves a model to disk
///
/// # Errors
///
/// Returns an error if serialization or file I/O fails
pub fn save_model(model: &Sequential, path: &Path) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut layer_configs = Vec::new();

    for layer in model.get_layers() {
        let config = if let Some(dense_layer) = layer.as_any().downcast_ref::<Dense>() {
            LayerConfig::Dense {
                input_size: dense_layer.input_size,
                output_size: dense_layer.output_size,
                weights: dense_layer.weights.clone(),
                bias: dense_layer.bias.clone(),
            }
        } else if let Some(conv2d_layer) = layer.as_any().downcast_ref::<Conv2D>() {
            LayerConfig::Conv2D {
                input_channels: conv2d_layer.input_channels,
                output_channels: conv2d_layer.output_channels,
                kernel_size: conv2d_layer.kernel_size,
                stride: conv2d_layer.stride,
                padding: conv2d_layer.padding,
                kernel: conv2d_layer.kernel.clone(),
                bias: conv2d_layer.bias.clone(),
            }
        } else if layer.as_any().downcast_ref::<ReLU>().is_some() {
            LayerConfig::ReLU
        } else if layer.as_any().downcast_ref::<Softmax>().is_some() {
            LayerConfig::Softmax
        } else if let Some(maxpool_layer) = layer.as_any().downcast_ref::<MaxPool2D>() {
            LayerConfig::MaxPool2D { kernel_size: maxpool_layer.kernel_size }
        } else if layer.as_any().downcast_ref::<Flatten>().is_some() {
            LayerConfig::Flatten
        } else if let Some(input_layer) = layer.as_any().downcast_ref::<Input>() {
            LayerConfig::Input(input_layer.clone())
        } else {
            return Err("Unknown layer type for serialization".into());
        };
        layer_configs.push(config);
    }

    let model_params = ModelParams { layers: layer_configs };
    let encoded: Vec<u8> = bincode::serialize(&model_params)?;
    let mut file = BufWriter::new(File::create(path)?);
    bincode::serialize_into(&mut file, &model_params)?;
    Ok(())
}

/// Loads a model from disk
///
/// # Errors
///
/// Returns an error if deserialization or file I/O fails
pub fn load_model(path: &Path) -> std::result::Result<Sequential, Box<dyn std::error::Error>> {
    let file = BufReader::new(File::open(path)?);
    let model_params: ModelParams = bincode::deserialize_from(file)?;

    let mut sequential = Sequential::new();
    for config in model_params.layers {
        let layer: Box<dyn Layer> = match config {
            LayerConfig::Input(input) => Box::new(input),
            LayerConfig::Dense { input_size, output_size, weights, bias } => {
                let mut dense = Dense::new(input_size, output_size, &mut rand::thread_rng()); // Dummy rng
                dense.weights = weights;
                dense.bias = bias;
                Box::new(dense)
            },
            LayerConfig::Conv2D { input_channels, output_channels, kernel_size, stride, padding, kernel, bias } => {
                let mut conv2d = Conv2D::new(input_channels, output_channels, kernel_size, stride, padding, &mut rand::thread_rng()); // Dummy rng
                conv2d.kernel = kernel;
                conv2d.bias = bias;
                Box::new(conv2d)
            },
            LayerConfig::ReLU => Box::new(ReLU::new()),
            LayerConfig::Softmax => Box::new(Softmax::new()),
            LayerConfig::MaxPool2D { kernel_size } => Box::new(MaxPool2D::new(kernel_size)),
            LayerConfig::Flatten => Box::new(Flatten::new()),
        };
        sequential.add_layer(layer);
    }

    Ok(sequential)
}