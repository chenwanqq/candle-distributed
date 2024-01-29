use super::dataset::Dataset;
use candle_core::Tensor;
use rand::{self, seq::SliceRandom, SeedableRng};

pub trait Sampler: Iterator<Item = Vec<Tensor>> + Send + Sync + Clone + 'static {
    fn output_tensor_num(&self) -> usize;
    fn reset(&mut self);
    fn get(&self, index: usize) -> Option<Vec<Tensor>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// SequentialSampler
#[derive(Clone)]
pub struct SequentialSampler<T>
where
    T: Dataset,
{
    dataset: T,
    index: usize,
}

impl<T> Sampler for SequentialSampler<T>
where
    T: Dataset,
{
    fn output_tensor_num(&self) -> usize {
        self.dataset.output_tensor_num()
    }
    fn reset(&mut self) {
        self.index = 0;
    }
    fn get(&self, index: usize) -> Option<Vec<Tensor>> {
        self.dataset.get(index)
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl<T> SequentialSampler<T>
where
    T: Dataset,
{
    pub fn new(dataset: T) -> Self {
        Self { dataset, index: 0 }
    }
}

impl<T> Iterator for SequentialSampler<T>
where
    T: Dataset,
{
    type Item = Vec<Tensor>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataset.len() {
            return None;
        }
        let res = self.dataset.get(self.index);
        self.index += 1;
        res
    }
}

/// RandomSampler
#[derive(Clone)]
pub struct RandomSampler<T>
where
    T: Dataset,
{
    dataset: T,
    index: usize,
    indexes: Vec<usize>,
    seed: Option<u64>,
    epoch: usize,
}

impl<T> RandomSampler<T>
where
    T: Dataset,
{
    pub fn new(dataset: T, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        let mut indexes: Vec<usize> = (0..dataset.len()).collect();
        indexes.shuffle(&mut rng);
        Self {
            dataset,
            index: 0,
            indexes,
            seed,
            epoch: 0,
        }
    }
}
impl<T> Sampler for RandomSampler<T>
where
    T: Dataset,
{
    fn output_tensor_num(&self) -> usize {
        self.dataset.output_tensor_num()
    }
    fn reset(&mut self) {
        let mut rng = match self.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed + self.epoch as u64),
            None => rand::rngs::StdRng::from_entropy(),
        };
        self.indexes.shuffle(&mut rng);
        self.index = 0;
        self.epoch += 1;
    }
    fn get(&self, index: usize) -> Option<Vec<Tensor>> {
        self.dataset.get(self.indexes[index])
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}
impl<T> Iterator for RandomSampler<T>
where
    T: Dataset,
{
    type Item = Vec<Tensor>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataset.len() {
            return None;
        }
        let res = self.dataset.get(self.indexes[self.index]);
        self.index += 1;
        res
    }
}
