use crate::matrix::matrix::Matrix;

#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    pub data: Vec<f64>,
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Vector {
        Vector { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, index: usize) -> f64 {
        self.data[index]
    }
    pub fn set(&mut self, index: usize, value: f64) {
        self.data[index] = value;
    }
}

impl From<Vector> for Matrix {
    fn from(vector: Vector) -> Self {
        let rows = vector.data.len();
        Matrix {
            data: vector.data,
            rows,
            cols: 1,
        }
    }
}

impl std::fmt::Display for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.data.len() {
            write!(f, "{} ", self.data[i])?;
        }
        Ok(())
    }
}
