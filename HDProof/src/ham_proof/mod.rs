/// Hamming distance proof module
///
/// This module implements a two-step proof system for verifying Hamming distance computations
/// between a target hash and a set of candidate hashes using multilinear extensions and
/// sumcheck protocols.
pub mod two_step_proof;

/// Bit decomposition based proof system
/// Utility functions for Hamming distance operations
pub use two_step_proof::HammingDistanceProof;
#[cfg(test)]
mod test;
/// tools functions
pub mod tools;
