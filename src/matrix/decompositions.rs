use crate::matrix::matrix::Matrix;
use crate::utils::error::Error;

pub fn lu(matrix: &Matrix) -> Result<(Matrix, Matrix), Error> {
    if matrix.rows != matrix.cols {
        return Err(Error::MatrixNotSquare);
    }
    return Ok((matrix.clone(), matrix.clone()));
}





pub fn svd(matrix: &Matrix) -> Result<(Matrix, Vec<f64>, Matrix), Error> {
    todo!()
}

pub fn qr(matrix: &Matrix) -> Result<(Matrix, Matrix), Error> {
    todo!()
}

pub fn eigen(matrix: &Matrix) -> Result<(Matrix, Matrix), Error> {
    todo!()
}