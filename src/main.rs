use data::dataset::Dataset;

mod data;


struct TestDataset {
    x: Vec<[u8;256]>,
    y: Vec<u8>,
}

impl Dataset for TestDataset {
    type Output = ([u8;256], u8);
    fn len(&self) -> usize {
        self.x.len()
    }
    fn get(&self, index: usize) -> Self::Output {
        (self.x[index], self.y[index])
    }
}

fn main() {
    println!("Hello, world!");
}
