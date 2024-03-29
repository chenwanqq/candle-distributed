use std::vec;
mod data_utils;

use candle_core::Tensor;
use data_utils::dataset::Dataset;

use data_utils::dataloader::DataLoader;

#[derive(Clone)]
struct TestDataset {
    x: Vec<[u8; 256]>,
    y: Vec<[u8; 2]>,
    device: candle_core::Device,
}

impl Dataset for TestDataset {
    fn get(&self, index: usize) -> Option<Vec<Tensor>> {
        if index > self.len() {
            return None;
        }
        let tensor0 =
            Tensor::from_raw_buffer(&self.x[index], candle_core::DType::F32, &[256], &self.device)
                .unwrap();
        let tensor1 =
            Tensor::from_raw_buffer(&self.y[index], candle_core::DType::U8, &[2], &self.device)
                .unwrap();
        Some(vec![tensor0, tensor1])
    }
    fn output_tensor_num(&self) -> usize {
        2
    }
    fn len(&self) -> usize {
        self.x.len()
    }
}

fn main() {
    let mut y = vec![[0_u8; 2]; 10];
    for (i, item) in y.iter_mut().enumerate().take(10) {
        item[0] = i as u8;
        item[1] = 9 - i as u8;
    }
    let dataset = TestDataset {
        x: vec![[0; 256]; 10],
        y,
        device: candle_core::Device::Cpu,
    };

    let mut dataloader = DataLoader::new_multi_worker(dataset, true, 3, false, None, 2, Some(2));

    for epoch in 0..2 {
        println!("epoch: {}", epoch);
        for (i, batch) in dataloader.by_ref().enumerate() {
            println!("batch: {}", i);
            for tensor in batch {
                println!("tensor: {:?}", tensor);
            }
        }
        dataloader.reset();
    }
}
