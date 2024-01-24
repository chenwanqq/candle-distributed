use candle_core::Tensor;

use super::dataset::Dataset;
use super::dataloader::DataLoader;

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
            Tensor::from_raw_buffer(&self.x[index], candle_core::DType::U8, &[256], &self.device)
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
        return self.x.len();
    }
}

#[test]
fn test_single_worker() {
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
    
        let mut dataloader = DataLoader::new_single_worker(
            dataset.clone(),
            false,
            3,
            false,
            None,
        );
        for epoch in 0..2 {
            println!("epoch: {}", epoch);
            for (i, batch) in dataloader.by_ref().enumerate() {
                let y = batch[1].to_vec2::<u8>().unwrap();
                println!("batch: {}, y: {:?}", i,y);
            }
            dataloader.reset()
        }

        let mut dataloader = DataLoader::new_single_worker(
            dataset.clone(),
            true,
            3,
            true,
            None,
        );
        for epoch in 0..2 {
            println!("epoch: {}", epoch);
            for (i, batch) in dataloader.by_ref().enumerate() {
                let y = batch[1].to_vec2::<u8>().unwrap();
                println!("batch: {}, y: {:?}",i, y);
            }
            dataloader.reset()
        }
}

#[test]
fn test_multi_worker() {
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

    let mut dataloader = DataLoader::new_multi_worker(
        dataset.clone(),
        false,
        3,
        false,
        None,
        5,
        Some(5),
    );
    
    for epoch in 0..2 {
        println!("epoch: {}", epoch);
        for (i, batch) in dataloader.by_ref().enumerate() {
            let y = batch[1].to_vec2::<u8>().unwrap();
            println!("batch: {}, y: {:?}",i, y);
        }
        dataloader.reset();
    }

    let mut dataloader = DataLoader::new_multi_worker(
        dataset,
        true,
        3,
        false,
        Some(333),
        5,
        Some(5),
    );
    for epoch in 0..2 {
        println!("epoch: {}", epoch);
        for (i, batch) in dataloader.by_ref().enumerate() {
            let y = batch[1].to_vec2::<u8>().unwrap();
            println!("batch: {}, y: {:?}",i, y);
        }
        dataloader.reset();
    }
}
