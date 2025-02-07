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
        let d = matrix.get(1, 1);

        let trace = a + d; // diagnal sum 2x2 only
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

pub fn inv(matrix: &Matrix) -> Option<Matrix> {
    assert_eq!(
        matrix.rows, matrix.cols,
        "Matrix must be square to compute inverse"
    );

    let n = matrix.rows;
    let mut augmented = Matrix {
        rows: n,
        cols: 2 * n,
        data: vec![0.0; n * 2 * n],
    };

    // init augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            augmented.data[i * augmented.cols + j] = matrix.data[i * matrix.cols + j];
        }
        augmented.data[i * augmented.cols + (i + n)] = 1.0; // identity matrix here bc using the function was a wast of space!
    }

    // Gauss-Jordan elim
    for i in 0..n {
        // find pivot row
        let mut pivot_row = i;
        for j in i + 1..n {
            if augmented.data[j * augmented.cols + i].abs()
                > augmented.data[pivot_row * augmented.cols + i].abs()
            {
                pivot_row = j;
            }
        }

        // swap rows if necessary
        if pivot_row != i {
            for j in 0..2 * n {
                augmented
                    .data
                    .swap(i * augmented.cols + j, pivot_row * augmented.cols + j);
            }
        }

        // check for singular matrix
        if augmented.data[i * augmented.cols + i] == 0.0 {
            return None; // matrix is singular, no inverse exists
        }

        // normalize pivot row
        let pivot = augmented.data[i * augmented.cols + i];
        for j in 0..2 * n {
            augmented.data[i * augmented.cols + j] /= pivot;
        }

        // eliminate all others
        for k in 0..n {
            if k != i {
                let factor = augmented.data[k * augmented.cols + i];
                for j in 0..2 * n {
                    augmented.data[k * augmented.cols + j] -=
                        factor * augmented.data[i * augmented.cols + j];
                }
            }
        }
    }

    // extract inverse matrix from the aug matrix
    let mut inverse_data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            // round to 6 decimal places
            inverse_data[i * n + j] =
                (augmented.data[i * augmented.cols + (j + n)] * 1e6).round() / 1e6;
        }
    }

    Some(Matrix {
        rows: n,
        cols: n,
        data: inverse_data,
    })
}
