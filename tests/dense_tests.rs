#![cfg_attr(rustfmt, rustfmt_skip)]

use linear_algebra_ml::matrix::matrix::Matrix;
use linear_algebra_ml::matrix::ops;


#[test]
fn test_matrix_new() {
    let matrix = Matrix::new(2, 3, vec![
        1.0, 2.0, 
        3.0, 4.0, 
        5.0, 6.0
    ]);
    assert_eq!(matrix.rows, 2);
    assert_eq!(matrix.cols, 3);
    assert_eq!(matrix.data, vec![
        1.0, 2.0, 
        3.0, 4.0, 
        5.0, 6.0
    ]);
}

#[test]
fn test_matrix_zeros() {
    let matrix = Matrix::zeros(2, 3);
    assert_eq!(matrix.rows, 2);
    assert_eq!(matrix.cols, 3);

    assert_eq!(matrix.data, vec![
        0.0, 0.0, 
        0.0, 0.0, 
        0.0, 0.0
    ]);
}

#[test]
fn test_matrix_identity() {
    let matrix = Matrix::identity(3);
    assert_eq!(matrix.rows, 3);
    assert_eq!(matrix.cols, 3);

    assert_eq!(matrix.data, vec![
        1.0, 0.0, 0.0, 
        0.0, 1.0, 0.0, 
        0.0, 0.0, 1.0
    ]);
}

#[test]
fn test_matrix_get() {
    let matrix = Matrix::new(3, 2, vec![
        1.0, 2.0, 
        3.0, 4.0, 
        5.0, 6.0
    ]);
    assert_eq!(matrix.get(0, 0), 1.0);
    assert_eq!(matrix.get(0, 1), 2.0);
    assert_eq!(matrix.get(1, 0), 3.0);
    assert_eq!(matrix.get(1, 1), 4.0);
}

#[test]
fn test_matrix_set() {
    let mut matrix = Matrix::zeros(2, 3);
    matrix.set(0, 0, 1.0);
    matrix.set(0, 1, 2.0);
    matrix.set(1, 0, 3.0);
    matrix.set(1, 1, 4.0);
    assert_eq!(matrix.get(0, 0), 1.0);
    assert_eq!(matrix.get(0, 1), 2.0);
    assert_eq!(matrix.get(1, 0), 3.0);
    assert_eq!(matrix.get(1, 1), 4.0);
}

#[test]
fn test_matrix_add() {
    let a = Matrix::new(2, 3, vec![
        1.0, 2.0, 
        3.0, 4.0, 
        5.0, 6.0
    ]);
    let b = Matrix::new(2, 3, vec![
        7.0, 8.0, 
        9.0, 10.0, 
        11.0, 12.0
    ]);
    let result = ops::add(&a, &b).unwrap();
    assert_eq!(result.rows, 2);
    assert_eq!(result.cols, 3);
    assert_eq!(result.data, vec![
        8.0, 10.0, 
        12.0, 14.0, 
        16.0, 18.0
    ]);
}

#[test]
fn test_matrix_mul() {
    let a = Matrix::new(2, 3, vec![
        1.0, 2.0, 3.0,  
        4.0, 5.0, 6.0   
    ]);

    let b = Matrix::new(3, 2, vec![
        7.0, 8.0,   
        9.0, 10.0,  
        11.0, 12.0  
    ]);

    let result = ops::mul(&a, &b).unwrap();

    let expected = Matrix::new(2, 2, vec![
        (1.0 * 7.0) + (2.0 * 9.0) + (3.0 * 11.0),  
        (1.0 * 8.0) + (2.0 * 10.0) + (3.0 * 12.0), 
        
        (4.0 * 7.0) + (5.0 * 9.0) + (6.0 * 11.0),  
        (4.0 * 8.0) + (5.0 * 10.0) + (6.0 * 12.0)  
    ]);

    assert_eq!(result, expected);
}

#[test]
fn test_matrix_sub() {
    let a = Matrix::new(2, 3, vec![
        1.0, 2.0, 3.0,  
        4.0, 5.0, 6.0   
    ]);

    let b = Matrix::new(2, 3, vec![
        7.0, 8.0, 9.0,  
        10.0, 11.0, 12.0   
    ]);

    let result = ops::sub(&a, &b).unwrap();

    let expected = Matrix::new(2, 3, vec![
        -6.0, -6.0, -6.0,  
        -6.0, -6.0, -6.0   
    ]);

    assert_eq!(result, expected);
}

#[test]
fn test_matrix_div() {
    let a = Matrix::new(2, 3, vec![
        1.0, 2.0, 3.0,  
        4.0, 5.0, 6.0   
    ]);

    let b = Matrix::new(2, 3, vec![
        7.0, 8.0, 9.0,  
        10.0, 11.0, 12.0   
    ]);

    let result = ops::div(&a, &b).unwrap();

    let expected = Matrix::new(2, 3, vec![
        (1.0 / 7.0), (2.0 / 8.0), (3.0 / 9.0),  
        (4.0 / 10.0), (5.0 / 11.0), (6.0 / 12.0)   
    ]);

    assert_eq!(result, expected);
}

#[test]
fn test_matrix_transpose() {
    let a = Matrix::new(2, 3, vec![
        1.0, 2.0, 3.0,  
        4.0, 5.0, 6.0   
    ]);

    let result = ops::transpose(&a);

    let expected = Matrix::new(3, 2, vec![
        1.0, 4.0,  
        2.0, 5.0,  
        3.0, 6.0   
    ]);

    assert_eq!(result, expected);
}

#[test]
fn test_matrix_scalar_mul() {
    let a = Matrix::new(2, 3, vec![
        1.0, 2.0, 3.0,  
        4.0, 5.0, 6.0   
    ]);

    let result = ops::scalar_mul(&a, 2.0);

    let expected = Matrix::new(2, 3, vec![
        2.0, 4.0, 6.0,  
        8.0, 10.0, 12.0   
    ]);

    assert_eq!(result, expected);
}