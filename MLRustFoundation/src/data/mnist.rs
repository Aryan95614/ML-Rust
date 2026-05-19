use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use flate2::read::GzDecoder;
use byteorder::{BigEndian, ReadBytesExt};
use ndarray::{ArrayD, IxDyn, s};
use crate::tensor::data::Tensor;

const BASE_URL: &str = "http://yann.lecun.com/exdb/mnist/";
const TRAIN_IMAGES_FILE: &str = "train-images-idx3-ubyte.gz";
const TRAIN_LABELS_FILE: &str = "train-labels-idx1-ubyte.gz";
const TEST_IMAGES_FILE: &str = "t10k-images-idx3-ubyte.gz";
const TEST_LABELS_FILE: &str = "t10k-labels-idx1-ubyte.gz";

const DATA_DIR: &str = "./data/mnist";

fn download_file(url: &str, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if path.exists() {
        println!("File already exists: {:?}", path);
        return Ok(());
    }

    println!("Downloading {} to {:?}", url, path);
    let mut response = reqwest::blocking::get(url)?;
    let mut file = File::create(path)?;
    io::copy(&mut response, &mut file)?;
    Ok(())
}

fn parse_idx_images(path: &Path) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut decoder = GzDecoder::new(file);
    let mut buffer = Vec::new();
    decoder.read_to_end(&mut buffer)?;

    let mut cursor = io::Cursor::new(buffer);

    let magic_number = cursor.read_u32::<BigEndian>()?;
    assert_eq!(magic_number, 2051, "Invalid magic number for image file");

    let num_images = cursor.read_u32::<BigEndian>()? as usize;
    let num_rows = cursor.read_u32::<BigEndian>()? as usize;
    let num_cols = cursor.read_u32::<BigEndian>()? as usize;

    let mut images = Vec::with_capacity(num_images * num_rows * num_cols);
    for _ in 0..(num_images * num_rows * num_cols) {
        images.push(cursor.read_u8()? as f32 / 255.0); // Normalize to [0, 1]
    }

    // Reshape to NCHW (num_images, 1, num_rows, num_cols)
    let array = ArrayD::from_shape_vec(IxDyn(&[num_images, 1, num_rows, num_cols]), images)?;
    Ok(array)
}

fn parse_idx_labels(path: &Path) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut decoder = GzDecoder::new(file);
    let mut buffer = Vec::new();
    decoder.read_to_end(&mut buffer)?;

    let mut cursor = io::Cursor::new(buffer);

    let magic_number = cursor.read_u32::<BigEndian>()?;
    assert_eq!(magic_number, 2049, "Invalid magic number for label file");

    let num_labels = cursor.read_u32::<BigEndian>()? as usize;

    let mut labels = Vec::with_capacity(num_labels);
    for _ in 0..num_labels {
        labels.push(cursor.read_u8()? as f32);
    }

    let array = ArrayD::from_shape_vec(IxDyn(&[num_labels]), labels)?;
    Ok(array)
}

pub fn load_mnist() -> Result<(Tensor, Tensor, Tensor, Tensor), Box<dyn std::error::Error>> {
    let data_path = PathBuf::from(DATA_DIR);
    fs::create_dir_all(&data_path)?;

    let train_images_path = data_path.join(TRAIN_IMAGES_FILE);
    let train_labels_path = data_path.join(TRAIN_LABELS_FILE);
    let test_images_path = data_path.join(TEST_IMAGES_FILE);
    let test_labels_path = data_path.join(TEST_LABELS_FILE);

    download_file(&format!("{}{}", BASE_URL, TRAIN_IMAGES_FILE), &train_images_path)?;
    download_file(&format!("{}{}", BASE_URL, TRAIN_LABELS_FILE), &train_labels_path)?;
    download_file(&format!("{}{}", BASE_URL, TEST_IMAGES_FILE), &test_images_path)?;
    download_file(&format!("{}{}", BASE_URL, TEST_LABELS_FILE), &test_labels_path)?;

    let train_images = parse_idx_images(&train_images_path)?;
    let train_labels = parse_idx_labels(&train_labels_path)?;
    let test_images = parse_idx_images(&test_images_path)?;
    let test_labels = parse_idx_labels(&test_labels_path)?;

    let train_images_shape = train_images.shape().to_vec();
    let train_labels_shape = train_labels.shape().to_vec();
    let test_images_shape = test_images.shape().to_vec();
    let test_labels_shape = test_labels.shape().to_vec();

    let train_images_tensor = Tensor::new(train_images, train_images_shape, false);
    let train_labels_tensor = Tensor::new(train_labels, train_labels_shape, false);
    let test_images_tensor = Tensor::new(test_images, test_images_shape, false);
    let test_labels_tensor = Tensor::new(test_labels, test_labels_shape, false);

    Ok((train_images_tensor, train_labels_tensor, test_images_tensor, test_labels_tensor))
}

pub struct MnistBatcher {
    images: Tensor,
    labels: Tensor,
    batch_size: usize,
    current_idx: usize,
}

impl MnistBatcher {
    pub fn new(images: Tensor, labels: Tensor, batch_size: usize) -> Self {
        assert_eq!(images.shape.0[0], labels.shape.0[0], "Images and labels must have the same number of samples");
        MnistBatcher {
            images,
            labels,
            batch_size,
            current_idx: 0,
        }
    }
}

impl Iterator for MnistBatcher {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.images.shape.0[0] {
            self.current_idx = 0; // Reset for next epoch
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.images.shape.0[0]);
        let batch_images_data = self.images.get_data().slice(s![self.current_idx..end_idx, .., .., ..]).to_owned().into_dyn();
        let batch_labels_data = self.labels.get_data().slice(s![self.current_idx..end_idx]).to_owned().into_dyn();

        let batch_images = Tensor::new(batch_images_data.clone(), batch_images_data.shape().to_vec(), false);
        let batch_labels = Tensor::new(batch_labels_data.clone(), batch_labels_data.shape().to_vec(), false);

        self.current_idx = end_idx;

        Some((batch_images, batch_labels))
    }
}

