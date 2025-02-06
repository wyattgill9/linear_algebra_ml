use linear_algebra_ml::dense::matrix::Matrix;
use linear_algebra_ml::*;
use linear_algebra_ml::sparse::matrix::SparseMatrix;

fn main() {
    // matrix mult test

    let a = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let b = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let c = dense::ops::mul(&a, &b).unwrap();
    println!("C: {:?}", c);

    let a = SparseMatrix::new(3, 3, vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0), (2, 0, 5.0), (2, 1, 6.0)]);
    let b = SparseMatrix::new(3, 3, vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0), (2, 0, 5.0), (2, 1, 6.0)]);
    let c = sparse::ops::mul(&a, &b).unwrap();
    println!("C: {:?}", c);
}