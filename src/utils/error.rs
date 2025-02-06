use std::error::Error as StdError;
use std::fmt;

#[derive(Debug)]
pub enum Error {
    MatrixSizeMismatch,
    MatrixNotSquare,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::MatrixSizeMismatch => write!(
                f,
                "Matrix size mismatch: matrices must have the same dimensions"
            ),
            Error::MatrixNotSquare => write!(f, "Matrix is not square"),
        }
    }
}

impl StdError for Error {}
