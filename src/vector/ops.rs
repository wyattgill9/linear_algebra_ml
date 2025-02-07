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
