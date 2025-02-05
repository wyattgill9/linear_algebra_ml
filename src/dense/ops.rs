use crate::dense::matrix::Matrix;
use crate::utils::error::Error;

pub fn add(a: &Matrix, b: &Matrix) -> Result<Matrix, Error> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(Error::MatrixSizeMismatch);
    }
    let mut result = Matrix::zeros(a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            result.set(i, j, a.get(i, j) + b.get(i, j));
        }
    }
    Ok(result)
}

// naive matrix mult
pub fn mul(a: &Matrix, b: &Matrix) -> Result<Matrix, Error> {
    if a.cols != b.rows {
        return Err(Error::MatrixSizeMismatch);
    }
    let mut result = Matrix::zeros(a.rows, b.cols);
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

pub fn sub(a: &Matrix, b: &Matrix) -> Result<Matrix, Error> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(Error::MatrixSizeMismatch);
    }
    let mut result = Matrix::zeros(a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            result.set(i, j, a.get(i, j) - b.get(i, j));
        }
    }
    Ok(result)
}

pub fn div(a: &Matrix, b: &Matrix) -> Result<Matrix, Error> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(Error::MatrixSizeMismatch);
    }
    let mut result = Matrix::zeros(a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            result.set(i, j, a.get(i, j) / b.get(i, j));
        }
    }
    Ok(result)
}

pub fn transpose(matrix: &Matrix) -> Matrix {
    let mut result = Matrix::zeros(matrix.cols, matrix.rows);
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            result.set(j, i, matrix.get(i, j));
        }
    }
    result
}