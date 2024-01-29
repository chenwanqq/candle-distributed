use candle_core::Tensor;

use super::{
    dataset::Dataset,
    sampler::{BatchSampler, RandomSampler, SingleWorkerBatchSampler},
};

pub struct DataLoader {
    batch_sampler: Box<dyn BatchSampler>,
}

impl DataLoader {
    pub fn new(batch_sampler: Box<dyn BatchSampler>) -> Self {
        Self { batch_sampler }
    }

    pub fn new_single_worker<T: Dataset>(
        dataset: T,
        shuffle: bool,
        batch_size: usize,
        drop_last: bool,
        seed: Option<u64>,
    ) -> Self {
        if shuffle {
            let sampler = RandomSampler::new(dataset, seed);
            let batch_sampler = SingleWorkerBatchSampler::new(sampler, batch_size, drop_last);
            Self::new(Box::new(batch_sampler))
        } else {
            let sampler = super::sampler::SequentialSampler::new(dataset);
            let batch_sampler = SingleWorkerBatchSampler::new(sampler, batch_size, drop_last);
            Self::new(Box::new(batch_sampler))
        }
    }

    pub fn new_multi_worker<T: Dataset>(
        dataset: T,
        shuffle: bool,
        batch_size: usize,
        drop_last: bool,
        seed: Option<u64>,
        num_workers: usize,
        prefetch_factor: Option<usize>,
    ) -> Self {
        if shuffle {
            let sampler = RandomSampler::new(dataset, seed);
            if let Some(prefetch_factor) = prefetch_factor {
                let batch_sampler = super::sampler::PrefetchMultiWorkerBatchSampler::new(
                    sampler,
                    batch_size,
                    drop_last,
                    num_workers,
                    prefetch_factor,
                );
                Self::new(Box::new(batch_sampler))
            } else {
                let batch_sampler = super::sampler::MultiWorkerBatchSampler::new(
                    sampler,
                    batch_size,
                    drop_last,
                    num_workers,
                );
                Self::new(Box::new(batch_sampler))
            }
        } else {
            let sampler = super::sampler::SequentialSampler::new(dataset);
            if let Some(prefetch_factor) = prefetch_factor {
                let batch_sampler = super::sampler::PrefetchMultiWorkerBatchSampler::new(
                    sampler,
                    batch_size,
                    drop_last,
                    num_workers,
                    prefetch_factor,
                );
                Self::new(Box::new(batch_sampler))
            } else {
                let batch_sampler = super::sampler::MultiWorkerBatchSampler::new(
                    sampler,
                    batch_size,
                    drop_last,
                    num_workers,
                );
                Self::new(Box::new(batch_sampler))
            }
        }
    }

    pub fn output_tensor_num(&self) -> usize {
        self.batch_sampler.output_tensor_num()
    }

    pub fn reset(&mut self) {
        self.batch_sampler.reset();
    }

    pub fn len(&self) -> usize {
        self.batch_sampler.len()
    }

    pub fn is_empty(&self) -> bool {
        self.batch_sampler.is_empty()
    }
}

impl Iterator for DataLoader {
    type Item = Vec<Tensor>;
    fn next(&mut self) -> Option<Self::Item> {
        self.batch_sampler.next()
    }
}
