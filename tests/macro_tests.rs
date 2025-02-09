#![cfg_attr(rustfmt, rustfmt_skip)]

use linear_algebra_ml::matrix::matrix::Matrix;
use linear_algebra_ml::{Sparse, Dense, Vector};

#[test]
fn test_sparse_matrix_macro() {
    let dense = Sparse!(3, 3, [
        0, 0, 1.0,
        1, 2, 5.0,
        2, 1, 3.0
    ]);

    let expected = Matrix {
        rows: 3,
        cols: 3,
        data: vec![
            1.0, 0.0, 0.0,
            0.0, 0.0, 5.0,
            0.0, 3.0, 0.0,
        ],
    };

    assert_eq!(dense.data, expected.data);
}


#[test]
fn test_dense_matrix_macro() {
    let dense = Dense!(3, 3, [
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    ]);

    let expected = Matrix {
        rows: 3,
        cols: 3,    
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    };

    assert_eq!(dense.data, expected.data);
}

#[test]
fn test_vector_macro() {
    let dense = Vector!(3, [
        1.0, 2.0, 3.0
    ]);

    let expected = Matrix {
        rows: 1,
        cols: 3,    
        data: vec![1.0, 2.0, 3.0],
    };

    assert_eq!(dense.data, expected.data);
}