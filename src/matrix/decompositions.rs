use crate::matrix::matrix::Matrix;
use crate::utils::error::Error;

pub fn lu(matrix: &Matrix) -> Result<(Matrix, Matrix), Error> {
    let n = matrix.rows;
    if n != matrix.cols {
        return Err(Error::MatrixNotSquare);
    }

    let mut lower = Matrix::identity(n);
    let mut upper = matrix.clone();

    for i in 0..n {
        if upper.get(i, i) == 0.0 {
            return Err(Error::SingularMatrix);
        }
        for j in i + 1..n {
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
