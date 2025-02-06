use crate::matrix::matrix::Matrix;
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

pub fn scalar_mul(matrix: &Matrix, scalar: f64) -> Matrix {
    let mut result = Matrix::zeros(matrix.rows, matrix.cols);
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            result.set(i, j, matrix.get(i, j) * scalar);
        }
    }
    result
}

pub fn power(matrix: &Matrix, scalar: f64) -> Matrix {
    let mut result = Matrix::zeros(matrix.rows, matrix.cols);
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            result.set(i, j, matrix.get(i, j).powf(scalar));
        }
    }
    result
}

pub fn determinant(matrix: &Matrix) -> f64 {
    if matrix.rows != matrix.cols {
        panic!("Matrix must be square");
    }

    let n = matrix.rows;

    if n == 1 {
        return matrix.get(0, 0);
    }

    if n == 2 {
        return matrix.get(0, 0) * matrix.get(1, 1) - matrix.get(0, 1) * matrix.get(1, 0);
    }

    let mut det = 0.0;
    for col in 0..n {
        let submatrix = matrix.minor(0, col);
        let sign = if col % 2 == 0 { 1.0 } else { -1.0 };
        det += sign * matrix.get(0, col) * determinant(&submatrix);
    }

    det
}

pub fn eigenvalues(matrix: &Matrix) -> Vec<f64> {
    if matrix.rows != matrix.cols {
        panic!("Matrix must be square");
    }

    if matrix.rows == 2 {
        let a = matrix.get(0, 0);
        let b = matrix.get(0, 1);
        let c = matrix.get(1, 0);
        let d = matrix.get(1, 1);

        let trace = a + d; // diagnal sum (2x2 only)
        let determinant = determinant(matrix);
        let discriminant = trace * trace - 4.0 * determinant;

        if discriminant >= 0.0 {
            let sqrt_discriminant = discriminant.sqrt();
            return vec![
                (trace + sqrt_discriminant) / 2.0,
                (trace - sqrt_discriminant) / 2.0,
            ];
        } else {
            panic!("Complex eigenvalues not supported");
        }
    }

    unimplemented!("Eigenvalue calculation for n > 2 is not implemented");
}
