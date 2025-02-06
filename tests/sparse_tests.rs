use linear_algebra_ml::sparse::matrix::SparseMatrix;
use linear_algebra_ml::sparse::ops;

#[test]
fn test_sparse_matrix_new() {
    let data = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
    let matrix = SparseMatrix::new(3, 3, data.clone());
    // println!("{}", matrix);
    assert_eq!(matrix.rows, 3);
    assert_eq!(matrix.cols, 3);
    assert_eq!(matrix.data, data);
}

#[test]
fn test_sparse_matrix_zeros() {
    let matrix = SparseMatrix::zeros(3, 3);
    assert_eq!(matrix.rows, 3);
    assert_eq!(matrix.cols, 3);
    assert_eq!(matrix.data.len(), 0);
}

#[test]
fn test_sparse_matrix_identity() {
    let matrix = SparseMatrix::identity(3);
    assert_eq!(matrix.rows, 3);
    assert_eq!(matrix.cols, 3);
    assert_eq!(matrix.data.len(), 3);
}

#[test]
fn test_sparse_matrix_get() {
    let data = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
    let matrix = SparseMatrix::new(3, 3, data.clone());
    assert_eq!(matrix.get(0, 0), 1.0);
    assert_eq!(matrix.get(1, 1), 2.0);
    assert_eq!(matrix.get(2, 2), 3.0);
}

#[test]
fn test_sparse_matrix_set() {
    let mut matrix = SparseMatrix::zeros(3, 3);
    matrix.set(0, 0, 1.0);
    matrix.set(1, 1, 2.0);
    matrix.set(2, 2, 3.0);
    assert_eq!(matrix.get(0, 0), 1.0);
    assert_eq!(matrix.get(1, 1), 2.0);
    assert_eq!(matrix.get(2, 2), 3.0);
}

#[test]
fn test_sparse_matrix_add() {
    let a_data = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
    let b_data = vec![(0, 0, 4.0), (1, 1, 5.0), (2, 2, 6.0)];
    let a = SparseMatrix::new(3, 3, a_data.clone());
    let b = SparseMatrix::new(3, 3, b_data.clone());
    let result = ops::add(&a, &b).unwrap();
    assert_eq!(result.get(0, 0), 5.0);
    assert_eq!(result.get(1, 1), 7.0);
    assert_eq!(result.get(2, 2), 9.0);
}

#[test]
fn test_sparse_matrix_mul() {
    let a_data = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
    let b_data = vec![(0, 0, 4.0), (1, 1, 5.0), (2, 2, 6.0)];
    let a = SparseMatrix::new(3, 3, a_data.clone());
    let b = SparseMatrix::new(3, 3, b_data.clone());
    let result = ops::mul(&a, &b).unwrap();
    assert_eq!(result.get(0, 0), 4.0);
    assert_eq!(result.get(1, 1), 10.0);
    assert_eq!(result.get(2, 2), 18.0);
}

#[test]
fn test_sparse_matrix_sub() {
    let a_data = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
    let b_data = vec![(0, 0, 4.0), (1, 1, 5.0), (2, 2, 6.0)];
    let a = SparseMatrix::new(3, 3, a_data.clone());
    let b = SparseMatrix::new(3, 3, b_data.clone());
    let result = ops::sub(&a, &b).unwrap();
    assert_eq!(result.get(0, 0), -3.0);
    assert_eq!(result.get(1, 1), -3.0);
    assert_eq!(result.get(2, 2), -3.0);
}

#[test]
fn test_sparse_matrix_div() {
    let a_data = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
    let b_data = vec![(0, 0, 4.0), (1, 1, 5.0), (2, 2, 6.0)];
    let a = SparseMatrix::new(3, 3, a_data.clone());
    let b = SparseMatrix::new(3, 3, b_data.clone());
    let result = ops::div(&a, &b).unwrap();
    assert_eq!(result.get(0, 0), 0.25);
    assert_eq!(result.get(1, 1), 0.4);
    assert_eq!(result.get(2, 2), 0.5);
}

#[test]
fn test_sparse_matrix_transpose() {
    let a_data = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
    let a = SparseMatrix::new(3, 3, a_data.clone());
    let result = ops::transpose(&a);

    assert_eq!(result.get(0, 0), 1.0);
    assert_eq!(result.get(0, 1), 0.0);
    assert_eq!(result.get(0, 2), 0.0);
    assert_eq!(result.get(1, 0), 0.0);
    assert_eq!(result.get(1, 1), 2.0);
    assert_eq!(result.get(1, 2), 0.0);
    assert_eq!(result.get(2, 0), 0.0);
    assert_eq!(result.get(2, 1), 0.0);
    assert_eq!(result.get(2, 2), 3.0);
}

#[test]
fn test_sparse_matrix_scalar_mul() {
    let a_data = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
    let a = SparseMatrix::new(3, 3, a_data.clone());
    let result = ops::scalar_mul(&a, 2.0);
    assert_eq!(result.get(0, 0), 2.0);
    assert_eq!(result.get(1, 1), 4.0);
    assert_eq!(result.get(2, 2), 6.0);
}

#[test]
pub fn test_sparse_matrix_power() {
    let a_data = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
    let a = SparseMatrix::new(3, 3, a_data.clone());
    let result = ops::power(&a, 2.0);
    assert_eq!(result.get(0, 0), 1.0);
    assert_eq!(result.get(1, 1), 4.0);
    assert_eq!(result.get(2, 2), 9.0);
}