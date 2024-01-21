use candle_core::Tensor;

pub trait Dataset: Send+Sync+Clone+'static {
    fn output_tensor_num(&self) -> usize;
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<Vec<Tensor>>;
}
