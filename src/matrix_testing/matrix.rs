use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Matrix {
    Dense {
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    },
    Sparse {
        rows: usize,
        cols: usize,
        data: HashMap<(usize, usize), f64>,
    },
}

impl Matrix {
    pub fn new_dense(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "Data length must match rows * cols");
        Matrix::Dense { rows, cols, data }
    }

    pub fn new_sparse(rows: usize, cols: usize, data: HashMap<(usize, usize), f64>) -> Self {
        Matrix::Sparse { rows, cols, data }
    }


    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix::Dense {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }


    pub fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Matrix::Dense {
            rows: n,
            cols: n,
            data,
        }
    }
    pub fn get(&self, row: usize, col: usize) -> f64 {
        match self {
            Matrix::Dense { rows, cols, data } => {
                assert!(row < *rows && col < *cols, "Index out of bounds");
                data[row * cols + col]
            }
            Matrix::Sparse { rows, cols, data } => {
                assert!(row < *rows && col < *cols, "Index out of bounds");
                *data.get(&(row, col)).unwrap_or(&0.0)
            }
        }
    }

    // Set the value at a specific position
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        match self {
            Matrix::Dense { rows, cols, data } => {
                assert!(row < *rows && col < *cols, "Index out of bounds");
                data[row * cols + col] = value;
            }
            Matrix::Sparse { rows, cols, data } => {
                assert!(row < *rows && col < *cols, "Index out of bounds");
                if value != 0.0 {
                    data.insert((row, col), value);
                } else {
                    data.remove(&(row, col));
                }
            }
        }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Matrix::Dense { rows, cols, data } => {
                for i in 0..*rows {
                    for j in 0..*cols {
                        write!(f, "{:.2} ", data[i * cols + j])?;
                    }
                    writeln!(f)?;
                }
            }
            Matrix::Sparse { rows, cols, data } => {
                for i in 0..*rows {
                    for j in 0..*cols {
                        let value = data.get(&(i, j)).unwrap_or(&0.0);
                        write!(f, "{:.2} ", value)?;
                    }
                    writeln!(f)?;
                }
            }
        }
        Ok(())
    }
}

fn main() {
    // Example usage
    let dense = Matrix::new_dense(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    println!("Dense Matrix:\n{}", dense);

    let mut sparse_data = HashMap::new();
    sparse_data.insert((0, 0), 1.0);
    sparse_data.insert((1, 1), 2.0);
    let sparse = Matrix::new_sparse(2, 2, sparse_data);
    println!("Sparse Matrix:\n{}", sparse);
}