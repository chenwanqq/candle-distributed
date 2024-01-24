use super::dataset::Dataset;
use candle_core::Tensor;
use futures::future::join_all;
use rand::{self, seq::SliceRandom, SeedableRng};
use std::sync::Arc;
use tokio::runtime::{self, Runtime};
use tokio::sync::{Mutex, RwLock};

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
        if self.index >= self.sampler.len() {
            return None;
        }
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
        let tmp = self.runtime.block_on(join_all(futures));
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
        self.index += self.batch_size;
        Some(res)
    }
}

pub struct PrefetchMultiWorkerBatchSampler<T>
where
    T: Sampler,
{
    sampler: Arc<RwLock<T>>,
    batch_size: usize,
    drop_last: bool,
    runtime: Runtime,
    num_workers: usize,
    index: usize,
    len: usize,
    prefetch_index: usize,
    prefetch_factor: usize,
    prefetch_queue: Arc<Mutex<Vec<Vec<Tensor>>>>,
    futures: Vec<tokio::task::JoinHandle<()>>,
}

impl<T> BatchSampler for PrefetchMultiWorkerBatchSampler<T>
where
    T: Sampler,
{
    fn output_tensor_num(&self) -> usize {
        self.runtime.block_on(async {
            let s = self.sampler.read().await;
            s.output_tensor_num()
        })
    }
    fn reset(&mut self) {
        //TODO: kill all task  in runtime and reset multiple states
        self.runtime.block_on(join_all(self.futures.drain(..)));
        let sampler_lock = self.sampler.clone();
        let prefetch_lock = self.prefetch_queue.clone();
        self.runtime.block_on(async move {
            let mut s = sampler_lock.write().await;
            s.reset();
            let mut p = prefetch_lock.lock().await;
            p.clear();
        });
        self.index = 0;
        self.prefetch_index = 0;
        //TODO: restart prefetch
        let mut futures = Vec::new();
        let mut prefetch_index = 0;
        while prefetch_index < self.prefetch_factor * self.batch_size && prefetch_index < self.len {
            let sampler_lock = self.sampler.clone();
            let prefetch_lock = self.prefetch_queue.clone();
            futures.push(self.runtime.spawn(async move {
                let s = sampler_lock.read().await;
                let v = s.get(prefetch_index);
                let mut p = prefetch_lock.lock().await;
                p.push(v.unwrap());
            }));
            prefetch_index += 1;
        }
        self.futures = futures;
        self.prefetch_index = prefetch_index;
    }
}

impl<T> PrefetchMultiWorkerBatchSampler<T>
where
    T: Sampler,
{
    pub fn new(
        sampler: T,
        batch_size: usize,
        drop_last: bool,
        num_workers: usize,
        prefetch_factor: usize,
    ) -> Self {
        let len = sampler.len();
        let runtime = runtime::Builder::new_multi_thread()
            .worker_threads(num_workers)
            .enable_time()
            .build()
            .unwrap();
        let sampler = Arc::new(RwLock::new(sampler));
        let prefetch_queue = Arc::new(Mutex::new(Vec::new()));
        let mut futures = Vec::new();
        let mut prefetch_index = 0;
        while prefetch_index < prefetch_factor * batch_size && prefetch_index < len {
            let sampler_lock = sampler.clone();
            let prefetch_lock = prefetch_queue.clone();
            futures.push(runtime.spawn(async move {
                let s = sampler_lock.read().await;
                let v = s.get(prefetch_index);
                let mut p = prefetch_lock.lock().await;
                p.push(v.unwrap());
            }));
            prefetch_index += 1;
        }
        Self {
            sampler,
            batch_size,
            drop_last,
            runtime: runtime,
            num_workers,
            index: 0,
            len,
            prefetch_index,
            prefetch_factor,
            prefetch_queue: prefetch_queue,
            futures,
        }
    }
}

impl<T> Iterator for PrefetchMultiWorkerBatchSampler<T>
where
    T: Sampler,
{
    type Item = Vec<Tensor>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len || (self.index + self.batch_size > self.len && self.drop_last) {
            return None;
        }
        let prefetch_lock = self.prefetch_queue.clone();
        let this_batch_size = std::cmp::min(self.batch_size, self.len - self.index);
        let tmp = self.runtime.block_on(async move {
            loop {
                let mut p = prefetch_lock.lock().await;
                if p.len() >= this_batch_size {
                    let mut res = Vec::new();
                    for _ in 0..this_batch_size {
                        res.push(p.remove(0));
                    }
                    return res;
                }
            }
        });
        self.index += self.batch_size;
        let mut prefetch_index = self.prefetch_index;
        while prefetch_index < self.len
            && prefetch_index < self.index + self.prefetch_factor * self.batch_size
        {
            let sampler_lock = self.sampler.clone();
            let prefetch_lock = self.prefetch_queue.clone();
            self.futures.push(self.runtime.spawn(async move {
                let s = sampler_lock.read().await;
                let v = s.get(prefetch_index);
                let mut p = prefetch_lock.lock().await;
                p.push(v.unwrap());
            }));
            prefetch_index += 1;
        }
        self.prefetch_index = prefetch_index;
        let output_tensor_num = self.output_tensor_num();
        let res: Vec<Tensor> = (0..output_tensor_num)
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
