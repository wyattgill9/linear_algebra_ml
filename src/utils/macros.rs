use crate::matrix::matrix::Matrix;
use crate::matrix::ops;

// converts a input : sparse matrix to a dense matrix
#[macro_export]
macro_rules! sparse_to_dense {
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
macro_rules! Matrix {
    ($rows:expr, $cols:expr, [$($row:expr),*]) => {
        {
            Matrix::new($rows, $cols, vec![$($row),*])
        }
    };
}


