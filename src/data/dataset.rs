use std::ops::{Index, IndexMut};
pub trait Dataset: {
    type Output;
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Self::Output;
}

