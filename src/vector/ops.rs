use crate::vector::vector::Vector;
use crate::utils::error::Error;

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
