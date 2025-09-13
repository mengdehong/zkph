//! Tool functions.
use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::CanonicalSerialize;
use ark_std::rc::Rc;
use rayon::prelude::*;
use std::cmp::min;
use std::fs::File;
use std::io::{BufRead, BufReader};

// Assuming crate::rng is correctly set up and accessible.
use crate::rng::{Blake2b512Rng, FeedableRNG};

/// Generates Poseidon configuration for Hyrax transcript (for testing purposes).
pub fn poseidon_config_for_test<F: PrimeField>() -> PoseidonConfig<F> {
    // Copied from 3rd/poly-commit poly-commit/src/lib.rs poseidon_parameters_for_test
    let full_rounds = 8;
    let partial_rounds = 31;
    let alpha = 17;

    // MDS matrix specific to Poseidon.
    let mds = vec![
        vec![F::one(), F::zero(), F::one()],
        vec![F::one(), F::one(), F::zero()],
        vec![F::zero(), F::one(), F::one()],
    ];

    // Generate arbitrary round constants using a deterministic RNG (Blake2b512Rng).
    let mut ark = Vec::new();
    let mut fs = Blake2b512Rng::setup(); // Setup the RNG.
    for _ in 0..(full_rounds + partial_rounds) {
        let mut res = Vec::new();
        for _ in 0..3 {
            res.push(F::rand(&mut fs)); // Generate random value for each round constant.
        }
        ark.push(res);
    }
    // Initialize PoseidonConfig with parameters. Rate is 2, capacity is 1.
    PoseidonConfig::new(full_rounds, partial_rounds, alpha, mds, ark, 2, 1)
}

/// Pads the MLE evaluation vector if the number of variables is odd.
/// This is done by duplicating the evaluation vector, effectively adding a "virtual variable" set to 0.
/// Returns the (potentially new) MLE and a boolean indicating if padding occurred.
pub fn pad_mle_to_even_vars<F: PrimeField>(
    mle: &Rc<DenseMultilinearExtension<F>>,
) -> (Rc<DenseMultilinearExtension<F>>, bool) {
    if mle.num_vars() % 2 == 0 {
        // If number of variables is already even, return the original MLE.
        (Rc::clone(mle), false)
    } else {
        // If odd, duplicate evaluations and increment the variable count.
        let mut new_evals = mle.evaluations.clone();
        new_evals.extend_from_slice(&mle.evaluations); // Append evaluations again.
        (
            Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                mle.num_vars() + 1, // Increment variable count.
                new_evals,
            )),
            true, // Indicate that padding occurred.
        )
    }
}

/// Appends a zero to the evaluation point if padding was applied to the MLE.
/// This ensures the evaluation point matches the (potentially padded) MLE's variable count.
pub fn pad_point<F: PrimeField>(point: &[F], padded: bool) -> Vec<F> {
    if padded {
        let mut padded_point = point.to_vec();
        padded_point.push(F::zero()); // Append zero for the virtual variable.
        padded_point
    } else {
        point.to_vec() // Return original point if no padding was applied.
    }
}

/// Reads a list of u64 values from a text file, where each line contains a hexadecimal number.
pub fn read_hash_file(file_path: &str) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    reader
        .lines()
        .map(|line| {
            let hex_str = line?.trim().to_string();
            if hex_str.is_empty() {
                return Ok(None); // Skip empty lines.
            }
            // Parse the hexadecimal string into a u64.
            u64::from_str_radix(&hex_str, 16)
                .map(Some) // Wrap the result in Some.
                .map_err(|e| e.into()) // Convert parsing error to Box<dyn Error>.
        })
        .filter_map(Result::transpose) // Filter out None values and propagate errors.
        .collect()
}

/// Expands a u64 hash into a 64-element bit vector (LSB first).
pub fn hash_to_bit_vector(hash: u64) -> Vec<u8> {
    (0..64).map(|i| ((hash >> i) & 1) as u8).collect()
}

/// Computes Hamming distances between vector `a` and each row in `b_matrix`.
pub fn compute_hamming_distances(a: &[u8], b_matrix: &[Vec<u8>]) -> Vec<u8> {
    b_matrix
        .par_iter()
        .map(|b_row| {
            // Hamming distance is the sum of bitwise XOR differences.
            a.iter()
                .zip(b_row.iter())
                .map(|(&a_b, &b_b)| a_b ^ b_b) // XOR corresponding bits.
                .sum() // Sum the differing bits.
        })
        .collect()
}

/// Computes the ceiling of log base 2 of n. This effectively gives the number of bits needed to represent n.
pub fn log2_ceil(n: usize) -> usize {
    if n == 0 {
        0
    } else {
        // For n > 0, (n-1).ilog2() gives floor(log2(n-1)). Adding 1 gives ceil(log2(n)).
        (n - 1).ilog2() as usize + 1
    }
}

/// Flattens a 0/1 matrix (representing B) into a single vector of length 2^(n+m).
/// The flattening is done in row-major order, interleaving `y` (m variables) and `x` (n variables).
/// The index for row `i` (y-value) and column `j` (x-value) is `i * 2^n + j`.
pub fn flatten_b_matrix(b_matrix: &[Vec<u8>], n: usize, m: usize) -> Vec<u8> {
    let total_size = 1 << (n + m); // Total size of the flattened vector.
    let mut result = vec![0u8; total_size];
    let rows = b_matrix.len();
    let cols = if rows > 0 { b_matrix[0].len() } else { 0 };
    // Determine the effective dimensions to process, considering potential padding/truncation.
    let max_rows = min(rows, 1 << m); // Number of rows to process, up to 2^m.
    let max_cols = min(cols, 1 << n); // Number of columns to process, up to 2^n.

    // Iterate through the provided matrix and fill the flattened vector.
    for (i, row) in b_matrix.iter().enumerate().take(max_rows) {
        for (j, &val) in row.iter().enumerate().take(max_cols) {
            let index = i * (1 << n) + j; // Calculate the flattened index.
            result[index] = val;
        }
    }
    result
}

/// Builds a histogram of Hamming distances (`d_vec_u8`) based on the lookup table `t_pad_u8`.
/// Counts the occurrences of each value in `d_vec_u8` and maps them to their positions in `t_pad_u8`.
/// `table_len` is the size of the lookup table (2^m_vars).
pub fn build_frequency_counts(d_vec_u8: &[u8], t_pad_u8: &[u8], table_len: usize) -> Vec<u64> {
    // Create a mapping from distance value to its first appearance index in the padded T table.
    // Initialize with usize::MAX to indicate "not found".
    let mut pos = [usize::MAX; 256];
    for (i, &v) in t_pad_u8.iter().enumerate() {
        if pos[v as usize] == usize::MAX {
            // If this value hasn't been mapped yet.
            pos[v as usize] = i; // Map value `v` to index `i`.
        }
    }

    // Use parallel processing to build the frequency counts.
    d_vec_u8
        .par_iter()
        .fold(
            // Initial accumulator: a vector of zeros for the histogram.
            || vec![0u64; table_len],
            // Folding function: update local counts based on distance value.
            |mut local, &d| {
                let mut idx = pos[d as usize]; // Find the index for distance `d`.
                if idx == usize::MAX {
                    // If `d` is not found in `t_pad_u8`, map it to index 0 (or a designated sentinel bin).
                    idx = 0;
                }
                local[idx] += 1u64; // Increment count for the corresponding index.
                local
            },
        )
        .reduce(
            // Initial value for reduction: a zero vector.
            || vec![0u64; table_len],
            // Reduction function: sum up local histograms into a global one.
            |mut acc, local| {
                for (a, b) in acc.iter_mut().zip(local.iter()) {
                    *a += *b; // Add counts from local histogram to accumulator.
                }
                acc
            },
        )
}

/// Feeds the first `k` serializable items from slice `v` into the provided `FeedableRNG`.
/// Used for generating transcript elements deterministically.
pub fn feed_prefix<FE: CanonicalSerialize>(
    fs: &mut impl FeedableRNG<Error = crate::Error>, // The RNG to feed.
    v: &[FE],                                        // Slice of items to feed.
    k: usize, // Number of items to feed from the beginning of the slice.
) -> Result<(), crate::Error> {
    let take = v.len().min(k); // Determine how many items to actually take.
    for item in v.iter().take(take) {
        fs.feed(item)?; // Feed each item into the RNG.
    }
    Ok(())
}
