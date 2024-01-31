use candle_core::Module;
use candle_distributed::data_utils::dataset::Dataset;
use candle_nn::{var_builder::SimpleBackend, Func, Optimizer, VarBuilder, VarMap};
use candle_transformers::models::resnet;
use criterion::criterion_main;
use indicatif::ProgressBar;
use std::path::Path;

//5 class, 1,2,3,4,5
#[derive(Clone)]
struct CompCarDataset {
    dataset_root: String,
    image_path_list: Vec<String>,
    label_list: Vec<u8>,
}

impl CompCarDataset {
    pub fn new(dataset_root: String, label: String) -> Self {
        let image_list_path = format!(
            "{}/train_test_split/classification/{}.txt",
            dataset_root, label
        );
        let content = std::fs::read_to_string(image_list_path).unwrap();
        let content_lines = content.split("\n").collect::<Vec<&str>>();
        let image_path_list: Vec<String> = content_lines
            .iter()
            .filter(|x| !x.is_empty())
            .map(|x| format!("{}/image/{}", dataset_root, x))
            .collect();
        println!("start loading labels");
        let pb = ProgressBar::new(image_path_list.len() as u64);
        let label_list: Vec<u8> = image_path_list
            .iter()
            .filter(|x| !x.is_empty())
            .map(|x| {
                let label_path = x.replace("image", "label").replace(".jpg", ".txt");
                let label_content = std::fs::read_to_string(label_path).unwrap();
                let label_str: String = label_content.lines().take(1).collect();
                pb.inc(1);
                label_str.parse::<u8>().unwrap()-1
            })
            .collect();
        pb.finish_with_message("load label done");
        Self {
            dataset_root,
            image_path_list,
            label_list,
        }
    }
}

fn image_normalization(
    image_tensor: &candle_core::Tensor,
) -> Result<candle_core::Tensor, candle_core::Error> {
    const IMAGE_NET_STD: [f32; 3] = [0.229, 0.224, 0.225];
    const IMAGE_NET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    image_tensor
        .broadcast_div(&candle_core::Tensor::from_vec(
            vec![255 as f32, 255 as f32, 255 as f32],
            &[1, 1, 3],
            &candle_core::Device::Cpu,
        )?)?
        .broadcast_sub(&candle_core::Tensor::from_vec(
            IMAGE_NET_MEAN.to_vec(),
            &[1, 1, 3],
            &candle_core::Device::Cpu,
        )?)?
        .broadcast_div(&candle_core::Tensor::from_vec(
            IMAGE_NET_STD.to_vec(),
            &[1, 1, 3],
            &candle_core::Device::Cpu,
        )?)
}

impl Dataset for CompCarDataset {
    fn output_tensor_num(&self) -> usize {
        2
    }

    fn len(&self) -> usize {
        self.image_path_list.len()
    }

    fn get(&self, index: usize) -> Option<Vec<candle_core::Tensor>> {
        //TODO: replace with image2, to provide normalization and f32

        let image_path = &self.image_path_list[index];
        let image = image::open(image_path);
        if image.is_err() {
            return None;
        }
        let image = image.unwrap();
        let image = image.resize_exact(224, 224, image::imageops::FilterType::Nearest);
        let image = image.to_rgb8();

        let image_tensor = candle_core::Tensor::from_raw_buffer(
            image.as_raw(),
            candle_core::DType::U8,
            &[224, 224, 3],
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .to_dtype(candle_core::DType::F32)
        .unwrap();
        let image_tensor = image_normalization(&image_tensor).unwrap();
        let label_tensor = candle_core::Tensor::from_raw_buffer(
            &[self.label_list[index]],
            candle_core::DType::U8,
            &[1],
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .to_dtype(candle_core::DType::U32)
        .unwrap();
        Some(vec![image_tensor, label_tensor])
    }
}

fn single_worker_benches() {
    let dataset_root = "/home/chenwanqq/candle-distributed/datasets/compcars".to_string();
    let dataset = CompCarDataset::new(dataset_root, "train".to_string());
    println!("dataset len: {}", dataset.len());
    let mut single_worker_dataloader =
        candle_distributed::data_utils::dataloader::DataLoader::new_single_worker(
            dataset, true, 64, false, None,
        );
    for epoches in 0..2 {
        println!("epoch: {}", epoches);
        let pb = ProgressBar::new(single_worker_dataloader.len() as u64);
        let start_time = std::time::Instant::now();
        let mut batch_time_0 = std::time::Instant::now();
        for (i, batch) in single_worker_dataloader.by_ref().enumerate() {
            let x = batch[0].permute((0, 3, 1, 2)).unwrap();
            let y = &batch[1];
            //sleep a while to simulate training
            std::thread::sleep(std::time::Duration::from_millis(200));
            pb.inc(1);
            let batch_time_1 = std::time::Instant::now();
            println!("batch time: {:?}", batch_time_1 - batch_time_0);
            batch_time_0 = batch_time_1;
        }
        let end_time = std::time::Instant::now();
        pb.finish_with_message("epoch done");
        println!("epoch time: {:?}", end_time - start_time);
        single_worker_dataloader.reset();
    }
}

struct TrainingArgs {
    learning_rate: f64,
    batch_size: usize,
    epochs: usize,
}

fn multi_worker_benches(args: &TrainingArgs) {
    let device = candle_core::Device::new_cuda(0).unwrap();
    //create dataset
    let dataset_root = "./datasets/compcars".to_string();
    let dataset = CompCarDataset::new(dataset_root, "train".to_string());
    println!("dataset len: {}", dataset.len());
    let mut dataloader = candle_distributed::data_utils::dataloader::DataLoader::new_multi_worker(
        dataset,
        true,
        args.batch_size,
        false,
        None,
        8,
        Some(2),
    );
    // create model
    let weight_path = Path::new("./weights/resnet18.safetensors");
    let mut weights = candle_core::safetensors::load(weight_path, &device).unwrap();
    let mut varmap = VarMap::new();
    for (k, v) in weights.iter() {
        varmap.set_one(k.to_string(), v);
    }
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model = resnet::resnet18(5, vb).unwrap();

    // create optimizer
    let adamw_params = candle_nn::ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), adamw_params).unwrap();

    // training loop
    for epoches in 0..args.epochs {
        println!("epoch: {}", epoches);
        let pb = ProgressBar::new(dataloader.len() as u64);
        let start_time = std::time::Instant::now();
        let mut batch_time_0 = std::time::Instant::now();
        let mut sum_loss = 0f32;
        for (i, batch) in dataloader.by_ref().enumerate() {
            let x = &batch[0].permute((0, 3, 1, 2)).unwrap().to_device(&device).unwrap();
            let batch_len = x.shape().dims()[0];
            println!("batch len: {}", batch_len);
            let y = &batch[1].reshape((batch_len,)).unwrap().to_device(&device).unwrap();
            let logits = model.forward(&x).unwrap();
            println!("logits shape: {:?}", logits.shape());
            println!("y shape: {:?}", y.shape());
            let loss = candle_nn::loss::cross_entropy(&logits, y).unwrap();
            optimizer.backward_step(&loss).unwrap();
            let this_loss = loss.to_vec0::<f32>().unwrap();
            sum_loss += this_loss;
            let batch_time_1 = std::time::Instant::now();
            println!(
                "batch: {:?}, loss: {:?}, batch time: {:?}",
                i,
                this_loss,
                batch_time_1 - batch_time_0,
            );
            batch_time_0 = batch_time_1;
            pb.inc(1);
        }
        let avg_loss = sum_loss / dataloader.len() as f32;
        let end_time = std::time::Instant::now();
        pb.finish_with_message("epoch done");
        println!(
            "epoch time: {:?} avg_loss: {:?}",
            end_time - start_time,
            avg_loss
        );
        dataloader.reset();
    }
}

fn main() {
    let args = TrainingArgs {
        learning_rate: 1e-3,
        batch_size: 32,
        epochs: 10,
    };
    multi_worker_benches(&args);
    //single_worker_benches()
}
//criterion_main!(benches);
