use std::ops::Sub;

use candle_core::Tensor;

fn print_2d_tensor(tensor: &Tensor) {
    let tensor: Vec<Vec<f32>> = tensor.to_vec2().unwrap();
    println!("{:?}", tensor);
}

fn print_3d_tensor(tensor: &Tensor) {
    let tensor: Vec<Vec<Vec<f32>>> = tensor.to_vec3().unwrap();
    for i in 0..tensor.len() {
        for j in 0..tensor[i].len() {
            println!("{}, {}, {:?}", i, j, tensor[i][j]);
        }
    }
}

#[test]
fn test_tensor_sub() {
    //create a (4,4,3) tensor to represent an image
    let tensor1 = Tensor::ones(
        (4, 4, 3),
        candle_core::DType::F32,
        &candle_core::Device::Cpu,
    )
    .unwrap();
    // create a (1,1,3) tensor to represent a mean to sub;
    let tensor2 = Tensor::from_vec(
        vec![1 as f32, 2 as f32, 3 as f32],
        (1, 1, 3),
        &candle_core::Device::Cpu,
    )
    .unwrap();
    // sub the mean from the image
    let tensor3 = tensor1.broadcast_sub(&tensor2).unwrap();
    print_3d_tensor(&tensor3);
}

#[test]
fn test_tensor_div() {
    let raw_vec = vec![120 as u8, 150 as u8, 200 as u8];
    let tensor1 = Tensor::from_raw_buffer(
        &raw_vec,
        candle_core::DType::U8,
        &[1,1,3],
        &candle_core::Device::Cpu,
    )
    .unwrap()
    .to_dtype(candle_core::DType::F32)
    .unwrap();
    let tensor2 = Tensor::from_vec(
        vec![123.15 as f32, 115.90 as f32, 103.06 as f32],
        (1, 1, 3),
        &candle_core::Device::Cpu,
    )
    .unwrap();
    let tensor3 = tensor1.broadcast_div(&tensor2).unwrap();
    print_3d_tensor(&tensor3);
}
