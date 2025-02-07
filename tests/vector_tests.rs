use linear_algebra_ml::vector::ops;
use linear_algebra_ml::vector::vector::Vector;
// use linear_algebra_ml::vector::decompositions;

#[test]
fn test_vector_add() {
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let b = Vector::new(vec![4.0, 5.0, 6.0]);
    let result = ops::add(&a, &b).unwrap();
    assert_eq!(result, Vector::new(vec![5.0, 7.0, 9.0]));
}

#[test]
fn test_vector_sub() {
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let b = Vector::new(vec![4.0, 5.0, 6.0]);
    let result = ops::sub(&a, &b).unwrap();
    assert_eq!(result, Vector::new(vec![-3.0, -3.0, -3.0]));
}

#[test]
fn test_vector_mul() {
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let b = Vector::new(vec![4.0, 5.0, 6.0]);
    let result = ops::mul(&a, &b).unwrap();
    assert_eq!(result, Vector::new(vec![4.0, 10.0, 18.0]));
}

#[test]
fn test_vector_div() {
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let b = Vector::new(vec![4.0, 5.0, 6.0]);
    let result = ops::div(&a, &b).unwrap();
    assert_eq!(result, Vector::new(vec![0.25, 0.4, 0.5]));
}

#[test]
fn test_vector_dot() {
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let b = Vector::new(vec![4.0, 5.0, 6.0]);
    let result = ops::dot(&a, &b).unwrap();
    assert_eq!(result, 32.0);
}

#[test]
fn test_vector_sum() {
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let result = ops::sum(&a);
    assert_eq!(result, 6.0);
}

#[test]
fn test_vector_norm() {
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let result = ops::norm(&a);
    assert_eq!(result, 3.7416573867739413);
}

#[test]
fn test_vector_hadamard() {
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let b = Vector::new(vec![4.0, 5.0, 6.0]);
    let result = ops::hadamard(&a, &b);
    assert_eq!(result, Vector::new(vec![4.0, 10.0, 18.0]));
}

#[test]
fn test_vector_concat() {
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let b = Vector::new(vec![4.0, 5.0, 6.0]);
    let result = ops::concat(&a, &b);
    assert_eq!(result, Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
}

#[test]
fn test_vector_scalar_mul() {
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let result = ops::vec_scalar_mul(&a, 2.0);
    assert_eq!(result, Vector::new(vec![2.0, 4.0, 6.0]));
}
