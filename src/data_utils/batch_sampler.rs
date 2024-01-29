use std::{collections::BinaryHeap, sync::Arc};

use candle_core::Tensor;
use futures::future::join_all;
use tokio::{
    runtime::{self, Runtime},
    sync::{Mutex, RwLock},
};

use super::sampler::Sampler;

/// BatchSampler
pub trait BatchSampler: Iterator<Item = Vec<Tensor>> {
    fn output_tensor_num(&self) -> usize;
    fn reset(&mut self);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
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
    fn len(&self) -> usize {
        if self.drop_last {
            self.sampler.len() / self.batch_size
        } else {
            (self.sampler.len() + self.batch_size - 1) / self.batch_size
        }
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
        if (tmp.len() < self.batch_size && self.drop_last) || tmp.is_empty() {
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
    fn len(&self) -> usize {
        if self.drop_last {
            self.sampler.len() / self.batch_size
        } else {
            (self.sampler.len() + self.batch_size - 1) / self.batch_size
        }
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
        let tmp = self.runtime.block_on(join_all(futures)); //join_all keeps order
        let tmp = tmp
            .iter()
            .filter_map(|x| x.as_ref().ok())
            .filter_map(|x| x.as_ref())
            .cloned()
            .collect::<Vec<Vec<Tensor>>>();
        if (tmp.len() < self.batch_size && self.drop_last) || tmp.is_empty() {
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

#[derive(Clone)]
struct IndexedTensorVec {
    pub tensors: Vec<Tensor>,
    pub index: usize,
}

impl Ord for IndexedTensorVec {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.index.cmp(&self.index)
    }
}

impl PartialOrd for IndexedTensorVec {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for IndexedTensorVec {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for IndexedTensorVec {}

pub struct PrefetchMultiWorkerBatchSampler<T>
where
    T: Sampler,
{
    sampler: Arc<RwLock<T>>,
    batch_size: usize,
    drop_last: bool,
    runtime: Runtime,
    index: usize,
    sample_nums: usize,
    prefetch_index: usize,
    prefetch_factor: usize,
    prefetch_queue: Arc<Mutex<BinaryHeap<IndexedTensorVec>>>,
    futures: Vec<tokio::task::JoinHandle<()>>,
    sample_limits: usize,
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
        self.futures.clear();
        let sampler_lock = self.sampler.clone();
        self.runtime.block_on(async move {
            let mut s = sampler_lock.write().await;
            s.reset();
        });
        self.index = 0;
        self.prefetch_index = 0;
        let mut prefetch_index = 0;
        while prefetch_index < self.prefetch_factor * self.batch_size
            && prefetch_index < self.sample_limits
        {
            let sampler_lock = self.sampler.clone();
            let prefetch_queue_thread = self.prefetch_queue.clone();
            self.futures.push(self.runtime.spawn(async move {
                prefetch_single(sampler_lock, prefetch_queue_thread, prefetch_index).await;
            }));
            prefetch_index += 1;
        }
        self.prefetch_index = prefetch_index;
    }
    fn len(&self) -> usize {
        if self.drop_last {
            self.sample_nums / self.batch_size
        } else {
            (self.sample_nums + self.batch_size - 1) / self.batch_size
        }
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
        let prefetch_queue = Arc::new(Mutex::new(BinaryHeap::new()));
        let sample_nums = sampler.len();
        let sample_limits = if drop_last {
            sample_nums / batch_size * batch_size
        } else {
            sample_nums
        };
        let runtime = runtime::Builder::new_multi_thread()
            .worker_threads(num_workers)
            .enable_time()
            .build()
            .unwrap();
        let batch_nums = if drop_last {
            sample_nums / batch_size
        } else {
            (sample_nums + batch_size - 1) / batch_size
        };
        let mut futures = Vec::with_capacity(batch_nums);
        let sampler = Arc::new(RwLock::new(sampler));
        let mut prefetch_index = 0;
        while prefetch_index < prefetch_factor * batch_size && prefetch_index < sample_limits {
            let sampler_lock = sampler.clone();
            let prefetch_queue_thread = prefetch_queue.clone();
            futures.push(runtime.spawn(async move {
                prefetch_single(sampler_lock, prefetch_queue_thread, prefetch_index).await;
            }));
            prefetch_index += 1;
        }
        Self {
            sampler,
            batch_size,
            drop_last,
            runtime,
            index: 0,
            sample_nums,
            prefetch_index,
            prefetch_factor,
            prefetch_queue,
            futures,
            sample_limits,
        }
    }
}

async fn prefetch_single<T>(
    sampler: Arc<RwLock<T>>,
    prefetch_queue: Arc<Mutex<BinaryHeap<IndexedTensorVec>>>,
    index: usize,
) where
    T: Sampler,
{
    let s = sampler.read().await;
    let v = s.get(index);
    if let Some(v) = v {
        let mut q = prefetch_queue.lock().await;
        q.push(IndexedTensorVec { tensors: v, index });
    }
}

impl<T> Iterator for PrefetchMultiWorkerBatchSampler<T>
where
    T: Sampler,
{
    type Item = Vec<Tensor>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.sample_nums
            || (self.index + self.batch_size > self.sample_nums && self.drop_last)
        {
            return None;
        }
        let this_batch_size = std::cmp::min(self.batch_size, self.sample_nums - self.index);
        let index_limit = self.index + this_batch_size;
        let mut now_index = self.index;
        let prefetch_queue_thread = self.prefetch_queue.clone();
        let tmp = self.runtime.block_on(async move {
            let mut res = Vec::with_capacity(this_batch_size);
            while now_index < index_limit {
                let mut q = prefetch_queue_thread.lock().await;
                let top_value = q.peek();
                if top_value.is_none() {
                    continue;
                }
                if top_value.unwrap().index == now_index {
                    let top_value = q.pop().unwrap();
                    res.push(top_value.tensors);
                    now_index += 1;
                }
            }
            res
        });
        self.index += this_batch_size;
        let mut prefetch_index = self.prefetch_index;
        while prefetch_index < self.sample_limits
            && prefetch_index < self.index + self.prefetch_factor * self.batch_size
        {
            let sampler_lock = self.sampler.clone();
            let prefetch_queue_thread = self.prefetch_queue.clone();
            self.futures.push(self.runtime.spawn(async move {
                prefetch_single(sampler_lock, prefetch_queue_thread, prefetch_index).await;
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
