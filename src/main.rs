use std::vec;

use candle_core::Tensor;
use data::{dataset::Dataset, sampler::BatchSampler};

mod data;

struct TestDataset {
    x: Vec<[u8; 256]>,
    y: Vec<[u8; 2]>,
    device: candle_core::Device,
}

impl Dataset for TestDataset {
    fn get(&self, index: usize) -> Vec<Tensor> {
        let tensor0 =
            Tensor::from_raw_buffer(&self.x[index], candle_core::DType::U8, &[256], &self.device)
                .unwrap();
        let tensor1 =
            Tensor::from_raw_buffer(&self.y[index], candle_core::DType::U8, &[2], &self.device)
                .unwrap();
        vec![tensor0, tensor1]
    }
    fn output_tensor_num(&self) -> usize {
        2
    }
    fn len(&self) -> usize {
        return self.x.len();
    }
}

fn main() {
    let mut y = vec![[0 as u8; 2]; 10];
    for i in 0..10 {
        y[i][0] = i as u8;
        y[i][1] = 9 - i as u8;
    }
    let dataset = TestDataset {
        x: vec![[0; 256]; 10],
        y,
        device: candle_core::Device::Cpu,
    };
    let dataloader = data::dataloader::DataLoader::new(dataset, 3, true, true);
    for x in dataloader {
        println!("{:?}", x);
    }
}
