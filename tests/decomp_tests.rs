#![cfg_attr(rustfmt, rustfmt_skip)]

use linear_algebra_ml::matrix::matrix::Matrix;
use linear_algebra_ml::matrix::ops;
use linear_algebra_ml::matrix::decompositions;

#[test]
fn test_lu() {
    let a = Matrix::new(3, 3, vec![
        2.0, 3.0, 1.0,
        4.0, 7.0, 3.0,
        6.0, 18.0, 5.0
    ]);

    match decompositions::lu(&a) {
        Ok((l, u)) => {
            println!("Lower matrix:\n{}", l);
            println!("Upper matrix:\n{}", u);
            let result = ops::mul(&l, &u).unwrap();
            println!("Result:\n{}", result);
            assert_eq!(a, result);
        }
        Err(e) => println!("LU decomposition failed: {:?}", e),
    }
}