use super::dataset::Dataset;
use candle_core::Tensor;
use futures::future::join_all;
use futures::FutureExt;
use rand::{self, seq::SliceRandom, SeedableRng};
use std::sync::Arc;
use tokio::runtime::{self, Runtime};
use tokio::sync::RwLock;

pub trait Sampler: Iterator<Item = Vec<Tensor>> + Send + Sync + Clone + 'static {
    fn output_tensor_num(&self) -> usize;
    fn reset(&mut self);
    fn get(&self, index: usize) -> Option<Vec<Tensor>>;
    fn len(&self) -> usize;
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
    }
    fn get(&self, index: usize) -> Option<Vec<Tensor>> {
        self.dataset.get(index)
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

/// BatchSampler
pub trait BatchSampler: Iterator<Item = Vec<Tensor>> {
    fn output_tensor_num(&self) -> usize;
    fn reset(&mut self);
}

/// SingleWorkerBatchSampler
pub struct SingleWorkerBatchSampler<T>
where
    T: Sampler,
{
    sampler: T,
    batch_size: usize,
    drop_last: bool,
    index: usize,
}

impl<T> SingleWorkerBatchSampler<T>
where
    T: Sampler,
{
    pub fn new(sampler: T, batch_size: usize, drop_last: bool) -> Self {
        Self {
            sampler,
            batch_size,
            drop_last,
            index: 0,
        }
    }
}
impl<T> BatchSampler for SingleWorkerBatchSampler<T>
where
    T: Sampler,
{
    fn output_tensor_num(&self) -> usize {
        self.sampler.output_tensor_num()
    }
    fn reset(&mut self) {
        self.sampler.reset();
    }
}
impl<T> Iterator for SingleWorkerBatchSampler<T>
where
    T: Sampler,
{
    type Item = Vec<Tensor>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index + self.batch_size > self.sampler.len() && self.drop_last {
            return None;
        }
        let n = self.sampler.output_tensor_num();
        let tmp: Vec<Vec<Tensor>> = self.sampler.by_ref().take(self.batch_size).collect();
        if (tmp.len() < self.batch_size && self.drop_last) || tmp.len() == 0 {
            return None;
        }
        let res: Vec<Tensor> = (0..n)
            .map(|i| {
                Tensor::stack(
                    &tmp.iter().map(|x| x[i].clone()).collect::<Vec<Tensor>>(),
                    0,
                )
                .unwrap()
            })
            .collect();
        Some(res)
    }
}

//TODO: multi worker batch sampler use tokio
pub struct MultiWorkerBatchSampler<T>
where
    T: Sampler,
{
    sampler: T,
    batch_size: usize,
    drop_last: bool,
    runtime: Runtime,
    num_workers: usize,
    index: usize,
}

impl<T> MultiWorkerBatchSampler<T>
where
    T: Sampler,
{
    pub fn new(sampler: T, batch_size: usize, drop_last: bool, num_workers: usize) -> Self {
        Self {
            sampler,
            batch_size,
            drop_last,
            runtime: runtime::Builder::new_multi_thread()
                .worker_threads(num_workers)
                .build()
                .unwrap(),
            num_workers,
            index: 0,
        }
    }
}

impl<T> BatchSampler for MultiWorkerBatchSampler<T>
where
    T: Sampler,
{
    fn output_tensor_num(&self) -> usize {
        self.sampler.output_tensor_num()
    }
    fn reset(&mut self) {
        self.sampler.reset();
        self.index = 0;
    }
}

impl<T> Iterator for MultiWorkerBatchSampler<T>
where
    T: Sampler,
{
    type Item = Vec<Tensor>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index + self.batch_size > self.sampler.len() && self.drop_last {
            return None;
        }
        let n = self.sampler.output_tensor_num();
        let sampler_lock = Arc::new(RwLock::new(self.sampler.clone()));
        let mut futures = Vec::new();
        for i in self.index..self.index + self.batch_size {
            if i >= self.sampler.len() {
                break;
            }
            let sampler_lock = sampler_lock.clone();
            futures.push(self.runtime.spawn(async move {
                let sampler = sampler_lock.read().await;
                sampler.get(i)
            }));
        }
        let tmp = self.runtime.block_on(futures::future::join_all(futures));
        let tmp = tmp
            .iter()
            .filter(|x| x.is_ok())
            .map(|x| x.as_ref().unwrap())
            .filter(|x| x.is_some())
            .map(|x| x.as_ref().unwrap())
            .map(|x| x.clone())
            .collect::<Vec<Vec<Tensor>>>();
        if (tmp.len() < self.batch_size && self.drop_last) || tmp.len() == 0 {
            return None;
        }
        let res: Vec<Tensor> = (0..n)
            .map(|i| {
                Tensor::stack(
                    &tmp.iter().map(|x| x[i].clone()).collect::<Vec<Tensor>>(),
                    0,
                )
                .unwrap()
            })
            .collect();
        if (res.len() < self.batch_size && self.drop_last) || res.len() == 0 {
            return None;
        }
        self.index += tmp.len();
        Some(res)
    }
}
