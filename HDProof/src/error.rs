use ark_std::fmt;

use ark_std::string::String;
use core::fmt::Formatter;
/// Error type for this crate
#[derive(fmt::Debug)]
pub enum Error {
    /// protocol rejects this proof
    Reject(Option<String>),
    /// IO Error
    IOError,
    /// Serialization Error
    SerializationError,
    /// Random Generator Error
    RNGError,
    /// Other caused by other operations
    OtherError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Self::OtherError(s) = self {
            f.write_str(s)
        } else {
            f.write_fmt(format_args!("{self:?}"))
        }
    }
}

impl ark_std::error::Error for Error {}

impl From<ark_std::io::Error> for Error {
    fn from(_: ark_std::io::Error) -> Self {
        Self::IOError
    }
}

impl From<ark_serialize::SerializationError> for Error {
    fn from(_: ark_serialize::SerializationError) -> Self {
        Self::SerializationError
    }
}
impl From<rand::Error> for Error {
    fn from(_: rand::Error) -> Self {
        Self::RNGError
    }
}

// Allow using `?` on functions returning ark-poly-commit errors
impl From<poly_commit::Error> for Error {
    fn from(e: poly_commit::Error) -> Self {
        Self::OtherError(format!("poly-commit error: {e:?}"))
    }
}

// Allow using `?` on functions returning ml_sumcheck errors
impl From<ml_sumcheck::Error> for Error {
    fn from(e: ml_sumcheck::Error) -> Self {
        match e {
            ml_sumcheck::Error::Reject(Some(msg)) => Self::Reject(Some(msg)),
            ml_sumcheck::Error::Reject(None) => Self::Reject(None),
            ml_sumcheck::Error::OtherError(msg) => Self::OtherError(msg),
        }
    }
}
