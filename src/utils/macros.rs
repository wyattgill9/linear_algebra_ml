use crate::matrix::matrix::Matrix;
use crate::matrix::*;

// converts a input : sparse matrix to a dense matrix
#[macro_export]
macro_rules! Sparse {
    ($rows:expr, $cols:expr, [$($row:expr, $col:expr, $val:expr),*]) => {
        {
            let mut data = vec![0.0; $rows * $cols];
            $(
                data[$row * $cols + $col] = $val;
            )*
            Matrix {
                rows: $rows,
                cols: $cols,
                data,
            }
        }
    };
}

#[macro_export]
macro_rules! Dense {
    ($rows:expr, $cols:expr, [$($row:expr),*]) => {
        {
            Matrix::new($rows, $cols, vec![$($row),*])
        }
    };
}

#[macro_export]
macro_rules! Vector {
    ($rows:expr, [$($row:expr),*]) => {
        {
            Matrix::new($rows, 1, vec![$($row),*])
        }
    };
}
