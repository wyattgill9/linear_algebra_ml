use crate::matrix::matrix::Matrix;
use crate::utils::error::Error;

pub fn lu(matrix: &Matrix) -> Result<(Matrix, Matrix), Error> {
    if matrix.rows != matrix.cols {
        return Err(Error::MatrixNotSquare);
    }

    let n = matrix.rows;
    let mut lower = Matrix::identity(n);
    let mut upper = matrix.clone();

    for i in 0..n {
        if upper.get(i, i).abs() < 1e-9 {
            return Err(Error::SingularMatrix);
        }

        for j in (i + 1)..n {
            let factor = upper.get(j, i) / upper.get(i, i);
            lower.set(j, i, factor);

            for k in i..n {
                let value = upper.get(j, k) - factor * upper.get(i, k);
                upper.set(j, k, value);
            }
        }
    }

    Ok((lower, upper))
}

pub fn svd(matrix: &Matrix) -> Result<(Matrix, Vec<f64>, Matrix), Error> {
    return Ok((matrix.clone(), vec![], matrix.clone()));
}

pub fn qr(matrix: &Matrix) -> Result<(Matrix, Matrix), Error> {
    return Ok((matrix.clone(), matrix.clone()));
}

pub fn eigen(matrix: &Matrix) -> Result<(Matrix, Matrix), Error> {
    return Ok((matrix.clone(), matrix.clone()));
}
