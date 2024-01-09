use candle_core::Tensor;

pub trait Dataset {
    fn get(&self, index: usize) -> Vec<Tensor>;
    fn output_tensor_num(&self) -> usize;
    fn len(&self) -> usize;
}
