#![cfg_attr(rustfmt, rustfmt_skip)]

use linear_algebra_ml::matrix::matrix::Matrix;
use linear_algebra_ml::matrix::ops;
use linear_algebra_ml::matrix::decompositions;


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

#[test]
fn test_matrix_power() {
    let a = Matrix::new(2, 3, vec![
        1.0, 2.0, 3.0,  
        4.0, 5.0, 6.0   
    ]);

    let result = ops::power(&a, 2.0);

    let expected = Matrix::new(2, 3, vec![
        1.0, 4.0, 9.0,  
        16.0, 25.0, 36.0   
    ]);

    assert_eq!(result, expected);
}

#[test]
fn test_matrix_inv() {
    let a = Matrix::new(2, 2, vec![
        1.0, 2.0,  
        3.0, 4.0   
    ]);

    let result = ops::inv(&a).unwrap();

    let expected = Matrix::new(2, 2, vec![
        -2.0, 1.0,  
        1.5, -0.5   
    ]);

    assert_eq!(result, expected);
}


#[test]
fn test_matrix_eigenvalues() {
    let a = Matrix::new(2, 2, vec![
        1.0, 2.0,  
        3.0, 4.0   
    ]);

    let result = ops::eigenvalues(&a);

    assert_eq!(result, vec![5.372281323269014, -0.3722813232690143]);
}

#[test]
fn test_matrix_det() {
    let a = Matrix::new(2, 2, vec![
        1.0, 2.0,  
        3.0, 4.0   
    ]);

    let result = ops::determinant(&a);

    assert_eq!(result, -2.0);
}


/* VECTOR TESTS */

#[test]
fn test_vector_dot() {
    let a = Matrix::new(2, 1, vec![
        1.0,  
        2.0   
    ]);

    let b = Matrix::new(1, 2, vec![
        3.0, 4.0   
    ]);

    let result = ops::dot(&a, &b);

    assert_eq!(result, 11.0);
}

#[test]
fn test_vec_dot_product() {
    let a = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]);
    let b = Matrix::new(3, 1, vec![4.0, 5.0, 6.0]);
    assert_eq!(ops::dot(&a, &b), 32.0);
}

#[test]
fn test_cross_product() {
    let a = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);
    let b = Matrix::new(3, 1, vec![4.0, 5.0, 6.0]);
    let expected = Matrix::new(3, 1, vec![-3.0, 6.0, -3.0]);
    assert_eq!(ops::cross(&a, &b), expected);
}

#[test]
fn test_projection() {
    let a = Matrix::new(1, 3, vec![3.0, 4.0, 0.0]);
    let b = Matrix::new(1, 3, vec![6.0, 8.0, 0.0]); 
    let expected = Matrix::new(1, 3, vec![3.0, 4.0, 0.0]);
    assert_eq!(ops::projection(&a, &b), expected);
}

#[test]
fn test_normalize() {
    let a = Matrix::new(1, 2, vec![3.0, 4.0]);
    let normalized = ops::normalize(&a);
    let expected = Matrix::new(1, 2, vec![0.6, 0.8]);
    for i in 0..normalized.data.len() {
        assert!((normalized.data[i] - expected.data[i]).abs() < 1e-6);
    }
}

#[test]
fn test_angle_between_vectors() {
    let a = Matrix::new(1, 3, vec![1.0, 0.0, 0.0]); 
    let b = Matrix::new(1, 3, vec![0.0, 1.0, 0.0]); 
    let angle = ops::angle(&a, &b);
    assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-6); 
}