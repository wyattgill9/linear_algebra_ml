use linear_algebra_ml::vector::vector::Vector;
use linear_algebra_ml::vector::ops;
// use linear_algebra_ml::vector::decompositions;

#[test]
fn test_vector_add() {
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let b = Vector::new(vec![4.0, 5.0, 6.0]);
    let result = ops::add(&a, &b).unwrap();
    assert_eq!(result, Vector::new(vec![5.0, 7.0, 9.0]));
}