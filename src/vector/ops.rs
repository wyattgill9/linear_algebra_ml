use crate::utils::error::Error;
use crate::vector::vector::Vector;

pub fn add(a: &Vector, b: &Vector) -> Result<Vector, Error> {
    if a.len() != b.len() {
        return Err(Error::MatrixSizeMismatch);
    }
    let mut result = Vector::new(vec![0.0; a.len()]);
    for i in 0..a.len() {
        result.set(i, a.get(i) + b.get(i));
    }
    Ok(result)
}

pub fn sub(a: &Vector, b: &Vector) -> Result<Vector, Error> {
    if a.len() != b.len() {
        return Err(Error::MatrixSizeMismatch);
    }
    let mut result = Vector::new(vec![0.0; a.len()]);
    for i in 0..a.len() {
        result.set(i, a.get(i) - b.get(i));
    }
    Ok(result)
}

pub fn mul(a: &Vector, b: &Vector) -> Result<Vector, Error> {
    if a.len() != b.len() {
        return Err(Error::MatrixSizeMismatch);
    }
    let mut result = Vector::new(vec![0.0; a.len()]);
    for i in 0..a.len() {
        result.set(i, a.get(i) * b.get(i));
    }
    Ok(result)
}

pub fn div(a: &Vector, b: &Vector) -> Result<Vector, Error> {
    if a.len() != b.len() {
        return Err(Error::MatrixSizeMismatch);
    }
    let mut result = Vector::new(vec![0.0; a.len()]);
    for i in 0..a.len() {
        result.set(i, a.get(i) / b.get(i));
    }
    Ok(result)
}

pub fn dot(a: &Vector, b: &Vector) -> Result<f64, Error> {
    if a.len() != b.len() {
        return Err(Error::MatrixSizeMismatch);
    }
    let mut result = 0.0;
    for i in 0..a.len() {
        result += a.get(i) * b.get(i);
    }
    Ok(result)
}

pub fn vec_scalar_mul(vector: &Vector, scalar: f64) -> Vector {
    let mut result = Vector::new(vec![0.0; vector.len()]);
    for i in 0..vector.len() {
        result.set(i, vector.get(i) * scalar);
    }
    result
}

pub fn norm(vector: &Vector) -> f64 {
    let mut result = 0.0;
    for i in 0..vector.len() {
        result += vector.get(i) * vector.get(i);
    }
    result.sqrt()
}

pub fn hadamard(a: &Vector, b: &Vector) -> Vector {
    let mut result = Vector::new(vec![0.0; a.len()]);
    for i in 0..a.len() {
        result.set(i, a.get(i) * b.get(i));
    }
    result
}

pub fn concat(a: &Vector, b: &Vector) -> Vector {
    let mut result = Vector::new(vec![0.0; a.len() + b.len()]);
    for i in 0..a.len() {
        result.set(i, a.get(i));
    }
    for i in 0..b.len() {
        result.set(i + a.len(), b.get(i));
    }
    result
}

pub fn sum(vector: &Vector) -> f64 {
    let mut result = 0.0;
    for i in 0..vector.len() {
        result += vector.get(i);
    }
    result
}
