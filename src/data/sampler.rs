use std::borrow::BorrowMut;

use super::dataset::Dataset;
use candle_core::Tensor;
use rand::{self, seq::SliceRandom, SeedableRng};

pub trait Sampler: Iterator<Item = Vec<Tensor>> {
    fn output_tensor_num(&self) -> usize;
}

//SequentialSampler
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
        Some(res)
    }
}

//RandomSampler
pub struct RandomSampler<T>
where
    T: Dataset,
{
    dataset: T,
    index: usize,
    rng: rand::rngs::StdRng,
    indexes: Vec<usize>,
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
            rng,
            indexes,
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
        Some(res)
    }
}

//BatchSampler

pub struct BatchSampler<T>
where
    T: Sampler,
{
    sampler: T,
    batch_size: usize,
    drop_last: bool,
}

impl<T> BatchSampler<T>
where
    T: Sampler,
{
    pub fn new(sampler: T, batch_size: usize, drop_last: bool) -> Self {
        Self {
            sampler,
            batch_size,
            drop_last,
        }
    }
}

impl<T> Sampler for BatchSampler<T>
where
    T: Sampler,
{
    fn output_tensor_num(&self) -> usize {
        self.sampler.output_tensor_num()
    }
}

impl<T> Iterator for BatchSampler<T>
where
    T: Sampler,
{
    type Item = Vec<Tensor>;
    fn next(&mut self) -> Option<Self::Item> {
        let n = self.sampler.output_tensor_num();
        let mut res = Vec::with_capacity(n);
        let tmp: Vec<Vec<Tensor>> = self.sampler.borrow_mut().take(self.batch_size).collect();
        if (tmp.len() < self.batch_size && self.drop_last) || tmp.len() == 0 {
            return None;
        }
        for i in 0..n {
            let mut tmp2 = Vec::with_capacity(self.batch_size);
            for j in 0..self.batch_size {
                tmp2.push(tmp[j][i].clone());
            }
            res.push(Tensor::stack(&tmp2, 0).unwrap());
        }
        Some(res)
    }
}

//TODO: multi worker batch sampler

//TODO: Distributed Sampler
