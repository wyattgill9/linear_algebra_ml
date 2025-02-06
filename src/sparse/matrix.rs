use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<(usize, usize, f64)>,
}

impl SparseMatrix {
    pub fn new(rows: usize, cols: usize, data: Vec<(usize, usize, f64)>) -> Self {
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: Vec::new(),
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut data = Vec::new();
        for i in 0..n {
            data.push((i, i, 1.0));
        }
        Self {
            rows: n,
            cols: n,
            data,
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data
            .iter()
            .find(|&&(r, c, _)| r == row && c == col)
            .map(|&(_, _, v)| v)
            .unwrap_or(0.0)
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data.push((row, col, value));
    }
}

impl fmt::Display for SparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut output = String::new();

        for i in 0..self.rows {
            for j in 0..self.cols {
                // fd element at i, j or default to 0.0
                let value = self
                    .data
                    .iter()
                    .find(|&&(r, c, _)| r == i && c == j)
                    .map(|&(_, _, v)| v)
                    .unwrap_or(0.0);

                output.push_str(&format!("{:6.2} ", value));
            }
            output.push('\n');
        }

        write!(f, "{}", output)
    }
}
