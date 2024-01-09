use candle_core::Tensor;

use crate::data::sampler::{BatchSampler, RandomSampler, Sampler, SequentialSampler};

use super::dataset::Dataset;
pub struct DataLoader {
    sampler: Box<dyn Sampler>,
}

impl DataLoader {
    pub fn from_sampler<S>(sampler: S) -> Self
    where
        S: Sampler + 'static,
    {
        Self {
            sampler: Box::new(sampler),
        }
    }

    //TODO pub fn new_distributed

    //TODO: add num_workers
    pub fn new<T>(dataset: T, batch_size: usize, shuffle: bool, drop_last: bool) -> Self
    where
        T: Dataset + 'static,
    {
        if shuffle {
            let sampler = RandomSampler::new(dataset, None);
            let sampler = BatchSampler::new(sampler, batch_size, drop_last);
            Self {
                sampler: Box::new(sampler),
            }
        } else {
            let sampler = SequentialSampler::new(dataset);
            let sampler = BatchSampler::new(sampler, batch_size, drop_last);
            Self {
                sampler: Box::new(sampler),
            }
        }
    }
}

impl Iterator for DataLoader {
    type Item = Vec<Tensor>;
    fn next(&mut self) -> Option<Self::Item> {
        self.sampler.next()
    }
}
