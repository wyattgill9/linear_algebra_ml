use crate::sparse::matrix::SparseMatrix;
use crate::utils::error::Error;

pub fn add(a: &SparseMatrix, b: &SparseMatrix) -> Result<SparseMatrix, Error> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(Error::MatrixSizeMismatch);
    }
    let mut result = SparseMatrix::zeros(a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            result.set(i, j, a.get(i, j) + b.get(i, j));
        }
    }
    Ok(result)
}

pub fn mul(a: &SparseMatrix, b: &SparseMatrix) -> Result<SparseMatrix, Error> {
    if a.cols != b.rows {
        return Err(Error::MatrixSizeMismatch);
    }
    let mut result = SparseMatrix::zeros(a.rows, b.cols);
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut sum = 0.0;
            for k in 0..a.cols {
                sum += a.get(i, k) * b.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
    Ok(result)
}

pub fn sub(a: &SparseMatrix, b: &SparseMatrix) -> Result<SparseMatrix, Error> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(Error::MatrixSizeMismatch);
    }
    let mut result = SparseMatrix::zeros(a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            result.set(i, j, a.get(i, j) - b.get(i, j));
        }
    }
    Ok(result)
}

pub fn div(a: &SparseMatrix, b: &SparseMatrix) -> Result<SparseMatrix, Error> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(Error::MatrixSizeMismatch);
    }
    let mut result = SparseMatrix::zeros(a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            result.set(i, j, a.get(i, j) / b.get(i, j));
        }
    }
    Ok(result)
}

pub fn transpose(matrix: &SparseMatrix) -> SparseMatrix {
    let mut result = SparseMatrix::zeros(matrix.cols, matrix.rows);
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            result.set(j, i, matrix.get(i, j));
        }
    }
    result
}

pub fn scalar_mul(matrix: &SparseMatrix, scalar: f64) -> SparseMatrix {
    let mut result = SparseMatrix::zeros(matrix.rows, matrix.cols);
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            result.set(i, j, matrix.get(i, j) * scalar);
        }
    }
    result
}

pub fn power(matrix: &SparseMatrix, scalar: f64) -> SparseMatrix {
    let mut result = SparseMatrix::zeros(matrix.rows, matrix.cols);
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            result.set(i, j, matrix.get(i, j).powf(scalar));
        }
    }
    result
}