//! Tool functions for the two-step Hamming distance proof.

use super::HammingDistanceProof; // Entry point for the two-step proof structure.
use ark_bls12_381::{Fr, G1Affine}; // Specific curve parameters for testing.
use ark_ff::UniformRand; // Trait for generating random field elements.
use ark_poly::MultilinearExtension; // MLE implementation.
use ark_serialize::CanonicalSerialize; // Trait for serialization.
use ark_std::rand::Rng; // RNG trait.
                        // use ark_std::rc::Rc; // Use Rc for shared ownership of MLEs.
use ark_std::test_rng; // Function to get a default RNG.
use std::fs::File; // File system operations.
use std::io::Write; // Write trait for file output.
use std::time::Instant; // For timing operations.

/// Creates a temporary file containing random 64-bit hashes, one per line in hexadecimal format.
/// This file serves as the input data `B` for the Hamming distance calculation.
fn create_test_hash_file(num_hashes: usize) -> String {
    let path = format!("/tmp/test_hashes_{num_hashes}.txt"); // Define file path in /tmp.
    let mut file = File::create(&path).unwrap(); // Create or truncate the file.
    let mut rng = test_rng(); // Get a test RNG instance.
    for _ in 0..num_hashes {
        let hash: u64 = rng.gen(); // Generate a random 64-bit hash.
        writeln!(file, "{hash:016X}").unwrap(); // Write the hash as a 16-char uppercase hex string.
    }
    path // Return the path to the created file.
}

// --- Test Functions ---

/// Tests the full two-step proof system end-to-end.
/// This includes witness generation, proof generation, and verification.
#[test]
fn test_two_step_full_proof_system() {
    println!("\n--- Testing the full two-step proof system (two_step_proof) ---");

    type F = Fr; // Field element type.
    type G = G1Affine; // Elliptic curve group type.

    // Use a moderate size to keep test execution time reasonable.
    let num_hashes = 1 << 20; // Corresponds to 2^20 hashes (approx 1 million).
    let hash_file_path = create_test_hash_file(num_hashes); // Create the input hash file.
    let a_hash = 0xAAAABBBBCCCCDDDDu64; // A fixed hash for 'a'.

    // 1. Initialize the two-step proof system.
    //    The range threshold `tau` is set to 0, covering the full possible range [0, 64].
    println!("[1/3] Initializing the two_step_proof system...");
    let start = Instant::now();
    let mut prover = HammingDistanceProof::<F, G>::new(a_hash, &hash_file_path, 0).unwrap();
    println!(
        "Initialization complete, time taken: {:?}. n_vars={}, m_vars={}",
        start.elapsed(),
        prover.a_mle.num_vars(), // Number of variables for 'a'.
        prover.d_mle.num_vars()  // Number of variables for 'd' and the lookup table.
    );

    // Generate random challenges and points required for proof generation.
    let mut rng = test_rng();
    let alpha: F = F::rand(&mut rng); // Phase 1 challenge.
    let gamma: F = F::rand(&mut rng); // Phase 2 challenge.
    let lambda: F = F::rand(&mut rng); // Phase 2 challenge.
    let m_vars = prover.d_mle.num_vars(); // Number of variables for y-domain points.
    let r_y: Vec<F> = (0..m_vars).map(|_| F::rand(&mut rng)).collect(); // Random point r for y-domain.
    let z: Vec<F> = (0..m_vars).map(|_| F::rand(&mut rng)).collect(); // Random point z for verification.

    // 2. Generate the full two-step proof.
    println!("\n[2/3] Generating the two-step proof...");
    let start = Instant::now();
    let proof = prover
        .generate_full_proof(alpha, gamma, lambda, &r_y, &z)
        .unwrap();
    println!(
        "Proof generation complete, time taken: {:?}",
        start.elapsed()
    );

    // 3. Verify the generated proof.
    println!("\n[3/3] Verifying the two-step proof...");
    let start = Instant::now();
    let ok = prover.verify(&proof).unwrap();
    println!(
        "Proof verification complete, time taken: {:?}",
        start.elapsed()
    );
    assert!(ok, "two_step_proof verification failed!"); // Assert that verification passes.
    println!("✅ Successfully completed the two-step proof system test.");
}

/// Measures and reports performance metrics for the Hamming distance proof system.
/// Includes timings for witness construction, proof generation, and verification.
#[test]
fn test_two_step_performance_metrics() {
    println!("\n--- Testing performance metrics for the Hamming distance proof system ---");

    type F = Fr;
    type G = G1Affine;

    // Set a moderate number of hashes for benchmarking.
    // 2^20 hashes (approx 1 million) provides a balance between test duration and result accuracy.
    let num_hashes = 1 << 20;
    let hash_file_path = create_test_hash_file(num_hashes);
    let a_hash = 0xAAAABBBBCCCCDDDDu64;

    // 1. Measure Witness Construction Time (Initialization + Data Processing).
    println!("[1/4] Measuring witness construction time...");
    let witness_start = Instant::now();
    // This call includes reading hashes, computing distances, creating MLEs, and setting up PCS keys.
    let mut prover = HammingDistanceProof::<F, G>::new(a_hash, &hash_file_path, 0).unwrap();
    let witness_time = witness_start.elapsed();
    println!("Witness construction time: {:?}", witness_time);

    // 2. Generate Challenges and Points.
    // These are needed for proof generation and are generated once to ensure consistency.
    let mut rng = test_rng();
    let alpha: F = F::rand(&mut rng);
    let gamma: F = F::rand(&mut rng);
    let lambda: F = F::rand(&mut rng);
    let m_vars = prover.d_mle.num_vars();
    let r_y: Vec<F> = (0..m_vars).map(|_| F::rand(&mut rng)).collect();
    let z: Vec<F> = (0..m_vars).map(|_| F::rand(&mut rng)).collect();

    // 3. Measure Proof Generation Time.
    println!("[2/4] Measuring proof generation time...");
    let proof_gen_start = Instant::now();
    // Generate the full proof using the previously constructed witness and generated challenges.
    let proof = prover
        .generate_full_proof(alpha, gamma, lambda, &r_y, &z)
        .unwrap();
    let proof_gen_time = proof_gen_start.elapsed();
    println!("Proof generation time: {:?}", proof_gen_time);

    // 4. Measure Verification Time.
    println!("[3/4] Measuring verification time...");
    let verify_start = Instant::now();
    // Verify the generated proof using the same prover instance (which now holds necessary info).
    let ok = prover.verify(&proof).unwrap();
    let verify_time = verify_start.elapsed();
    println!("Verification time: {:?}", verify_time);
    assert!(ok, "Verification failed!"); // Ensure verification succeeds.

    // 5. Summarize results.
    println!("\n--- Hamming Distance Proof System Performance Metrics Summary ---");
    println!("Witness construction time: {:?}", witness_time);
    println!("Proof generation time: {:?}", proof_gen_time);
    println!("Verification time: {:?}", verify_time);
    println!("✅ Two-step proof performance test completed.");
}

/// Generates a table of performance metrics (Witness, Prove, Verify times, Proof Size)
/// for different input sizes (number of hashes).
#[test]
fn bench_two_step_proof_table() {
    // Define the input sizes (number of hashes) to benchmark.
    // Powers of 2 from 2^14 to 2^20 are common for scalability testing.
    let sizes: [usize; 4] = [1 << 14, 1 << 16, 1 << 18, 1 << 20];

    type F = Fr;
    type G = G1Affine;

    println!("\n--- Benchmarking table for the two-step proof system ---");
    // Print table header.
    println!("Size     | Witness(s) | Prove(s)   | Verify(s)  | Proof Size (bytes)");
    println!("---------|------------|------------|------------|--------------------");

    // Iterate through each size to benchmark.
    for &num_hashes in &sizes {
        let hash_file_path = create_test_hash_file(num_hashes); // Create hash file for the current size.
        let a_hash = 0xAAAABBBBCCCCDDDDu64; // Fixed hash for 'a'.

        // Measure Witness construction time.
        let witness_start = Instant::now();
        let mut prover = HammingDistanceProof::<F, G>::new(a_hash, &hash_file_path, 0).unwrap();
        let witness_time = witness_start.elapsed();

        // Generate challenges and points for proof generation.
        let mut rng = test_rng();
        let alpha: F = F::rand(&mut rng);
        let gamma: F = F::rand(&mut rng);
        let lambda: F = F::rand(&mut rng);
        let m_vars = prover.d_mle.num_vars();
        let r_y: Vec<F> = (0..m_vars).map(|_| F::rand(&mut rng)).collect();
        let z: Vec<F> = (0..m_vars).map(|_| F::rand(&mut rng)).collect();

        // Measure Proof Generation time.
        let prove_start = Instant::now();
        let proof = prover
            .generate_full_proof(alpha, gamma, lambda, &r_y, &z)
            .unwrap();
        let prove_time = prove_start.elapsed();

        // Calculate Proof Size.
        let mut proof_bytes = Vec::new();
        // Serialize the proof to get its size in bytes.
        proof.serialize_uncompressed(&mut proof_bytes).unwrap();
        let proof_size = proof_bytes.len();

        // Measure Verification time.
        let verify_start = Instant::now();
        let ok = prover.verify(&proof).unwrap();
        let verify_time = verify_start.elapsed();
        assert!(ok, "Verification failed!"); // Assert verification success.

        // Print the results for the current size in a formatted table row.
        println!(
            "2^{:>2} | {:>10.3}s | {:>8.3}s | {:>9.3}s | {:>18}",
            num_hashes.trailing_zeros(), // Display exponent (p) for size 2^p.
            witness_time.as_secs_f64(),
            prove_time.as_secs_f64(),
            verify_time.as_secs_f64(),
            proof_size
        );
    }
}
