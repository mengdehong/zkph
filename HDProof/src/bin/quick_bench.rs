use ark_bls12_381::{Fr, G1Affine};
use ark_ff::UniformRand;
use ark_poly::MultilinearExtension;
use ark_std::rand::Rng;
use hamproof::ham_proof::HammingDistanceProof;
use std::env;
use std::time::{Duration, Instant};

// ----------------------------- Configuration Parsing and Utilities -----------------------------

/// Configuration struct for benchmarking parameters.
#[derive(Debug, Clone)]
struct BenchCfg {
    /// Sizes to benchmark (number of hashes, 2^p).
    sizes: Vec<usize>,
    /// Number of warmup runs before taking measurements.
    warmup: usize,
    /// Number of samples to collect for statistics.
    samples: usize,
    /// Seed for reproducible randomness.
    seed: Option<u64>,
    /// Whether to use /dev/shm (tmpfs) for hash file storage.
    use_tmpfs: bool,
    /// Compatibility parameter for repeats (older versions might use this).
    repeats_compat: usize,
}

/// Creates a test hash file with a specified number of hashes.
/// Recreates the file if it doesn't exist or seems corrupted/incorrectly sized.
/// Uses tmpfs if available and configured.
fn create_test_hash_file(num_hashes: usize, cfg: &BenchCfg) -> String {
    // Determine the base directory for the hash file.
    let dir_base = if cfg.use_tmpfs && std::path::Path::new("/dev/shm").is_dir() {
        "/dev/shm" // Use shared memory filesystem if available and enabled.
    } else {
        "/tmp" // Default to /tmp directory.
    };
    // Construct the file path.
    let path = format!("{}/quick_bench_hashes_{}.txt", dir_base, num_hashes);

    // Check if the file needs to be recreated.
    let need_recreate = match std::fs::metadata(&path) {
        Ok(meta) => {
            // Estimate expected file size (num_hashes * (16 hex chars + newline)).
            let expected = (num_hashes as u64) * 17;
            let sz = meta.len();
            // Recreate if size is significantly smaller or larger than expected.
            sz < expected / 2 || sz > expected * 2
        }
        Err(_) => true, // Recreate if the file doesn't exist.
    };

    if need_recreate {
        // Use a seeded RNG for reproducibility if a seed is provided.
        use rand::SeedableRng;
        let mut file = std::fs::File::create(&path).unwrap();
        let mut rng = if let Some(seed) = cfg.seed {
            // Seed with provided value XORed with num_hashes for variation.
            rand::rngs::StdRng::seed_from_u64(seed ^ (num_hashes as u64))
        } else {
            // Use system entropy if no seed is provided.
            rand::rngs::StdRng::from_entropy()
        };
        // Write `num_hashes` random hashes to the file.
        for _ in 0..num_hashes {
            let hash: u64 = rng.gen();
            use std::io::Write;
            writeln!(file, "{hash:016X}").unwrap(); // Write hash in uppercase hex format.
        }
    }
    path // Return the path to the hash file.
}

/// Parses command-line arguments to configure benchmarking parameters.
fn parse_cfg() -> BenchCfg {
    let mut sizes: Option<Vec<usize>> = None;
    let mut warmup: Option<usize> = None;
    let mut samples: Option<usize> = None;
    let mut seed: Option<u64> = None;
    let mut use_tmpfs = false;
    let mut repeats: Option<usize> = None;

    // Iterate through command-line arguments.
    for arg in env::args() {
        if let Some(rest) = arg.strip_prefix("--sizes=") {
            // Parse --sizes=p1,p2,... where sizes are 2^p.
            let mut v = Vec::new();
            for s in rest.split(',') {
                if let Ok(p) = s.trim().parse::<u32>() {
                    // Only accept valid bit sizes (e.g., 10 to 32).
                    if (10..=32).contains(&p) {
                        v.push(1usize << p); // Store size as 2^p.
                    }
                }
            }
            if !v.is_empty() {
                sizes = Some(v);
            }
        } else if let Some(rest) = arg.strip_prefix("--warmup=") {
            // Parse --warmup=N.
            if let Ok(n) = rest.parse::<usize>() {
                warmup = Some(n);
            }
        } else if let Some(rest) = arg.strip_prefix("--samples=") {
            // Parse --samples=N.
            if let Ok(n) = rest.parse::<usize>() {
                samples = Some(n);
            }
        } else if let Some(rest) = arg.strip_prefix("--seed=") {
            // Parse --seed=S.
            if let Ok(n) = rest.parse::<u64>() {
                seed = Some(n);
            }
        } else if arg == "--tmpfs" || arg == "--memfs" {
            // Enable tmpfs usage.
            use_tmpfs = true;
        } else if let Some(rest) = arg.strip_prefix("--repeats=") {
            // Compatibility for older 'repeats' parameter.
            if let Ok(n) = rest.parse::<usize>() {
                repeats = Some(n);
            }
        }
    }

    // Determine final values for samples and repeats.
    // If 'samples' is explicitly given, use it. Otherwise, fall back to 'repeats' (defaulting to 3).
    let samples_val = samples.unwrap_or_else(|| repeats.unwrap_or(3)).max(1);
    let repeats_val = repeats.unwrap_or(3).max(1); // Default repeats if not specified.

    BenchCfg {
        // Default sizes if not provided: 2^14, 2^16, 2^18.
        sizes: sizes.unwrap_or_else(|| vec![1 << 14, 1 << 16, 1 << 18]),
        warmup: warmup.unwrap_or(1), // Default warmup: 1 run.
        samples: samples_val,
        seed,
        use_tmpfs,
        repeats_compat: repeats_val,
    }
}

/// Statistics calculated from a collection of durations.
#[derive(Debug, Clone)]
struct Stats {
    mean: f64,   // Average duration.
    median: f64, // Middle value after sorting.
    stddev: f64, // Standard deviation.
    p90: f64,    // 90th percentile.
    min: f64,    // Minimum duration.
    max: f64,    // Maximum duration.
}

/// Computes statistical measures (mean, median, stddev, p90, min, max) from a slice of durations.
fn compute_stats(durs: &[Duration]) -> Stats {
    let n = durs.len() as f64;
    // Convert durations to seconds as f64.
    let mut arr: Vec<f64> = durs.iter().map(|d| d.as_secs_f64()).collect();
    arr.sort_by(|a, b| a.partial_cmp(b).unwrap()); // Sort durations in ascending order.

    let sum: f64 = arr.iter().sum();
    let mean = sum / n.max(1.0); // Calculate mean, handle empty slice case.

    // Calculate median.
    let median = if arr.is_empty() {
        0.0
    } else if arr.len() % 2 == 1 {
        arr[arr.len() / 2] // Odd number of elements.
    } else {
        (arr[arr.len() / 2 - 1] + arr[arr.len() / 2]) / 2.0 // Even number of elements.
    };

    // Calculate variance and standard deviation.
    let var = if arr.len() > 1 {
        // Sample variance formula: sum((x - mean)^2) / (n - 1)
        arr.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)
    } else {
        0.0 // Variance is 0 if only one or zero samples.
    };
    let stddev = var.sqrt();

    // Calculate 90th percentile.
    let p90 = if arr.is_empty() {
        0.0
    } else {
        // Index for p90: round((n-1) * 0.90).
        let idx = ((arr.len() - 1) as f64 * 0.90).round() as usize;
        arr[idx]
    };

    Stats {
        mean,
        median,
        stddev,
        p90,
        min: *arr.first().unwrap_or(&0.0), // Get minimum, default to 0.0 if empty.
        max: *arr.last().unwrap_or(&0.0),  // Get maximum, default to 0.0 if empty.
    }
}

/// Formats statistics into a human-readable string.
fn fmt_stats(label: &str, s: &Stats) -> String {
    format!(
        "{label}: mean={:.6}s median={:.6}s p90={:.6}s std={:.6}s min={:.6}s max={:.6}s",
        s.mean, s.median, s.p90, s.stddev, s.min, s.max
    )
}

/// Measures the execution time of a closure `f` over multiple runs (warmup + samples).
/// Returns the duration of the first run and a vector of durations for the sampled runs.
fn measure_many<F: FnMut()>(warmup: usize, samples: usize, mut f: F) -> (Duration, Vec<Duration>) {
    // Measure the first run separately.
    let t0 = Instant::now();
    f();
    let first = t0.elapsed();

    // Perform warmup runs to stabilize performance (e.g., cache warm-up, JIT compilation).
    // Subtract 1 from warmup because the first run is already done.
    for _ in 0..warmup.saturating_sub(1) {
        f();
    }

    // Collect sampled measurements.
    let mut out = Vec::with_capacity(samples);
    for _ in 0..samples {
        let t = Instant::now();
        f();
        out.push(t.elapsed()); // Store the duration of this run.
    }
    (first, out) // Return first run duration and sampled durations.
}

fn main() {
    // Define the field and elliptic curve group for the benchmark.
    type F = Fr;
    type G = G1Affine;

    // Parse configuration from command-line arguments.
    let cfg = parse_cfg();
    // Print configuration being used.
    println!(
        "quick bench: sizes={:?}, warmup={}, samples={}, seed={:?}, tmpfs={} (repeats compat={})",
        cfg.sizes
            .iter()
            // Display bit sizes (p) instead of 2^p for clarity.
            .map(|n| n.trailing_zeros())
            .collect::<Vec<_>>(),
        cfg.warmup,
        cfg.samples,
        cfg.seed,
        cfg.use_tmpfs,
        cfg.repeats_compat
    );

    // Iterate through the specified sizes (number of hashes).
    for &num_hashes in &cfg.sizes {
        // Create an identifier for the current size (e.g., "2^14").
        let id = format!("2^{}", num_hashes.trailing_zeros());
        // Create the hash file needed for the proof system.
        let hash_file_path = create_test_hash_file(num_hashes, &cfg);
        // Fixed hash for 'a' for consistency across runs.
        let a_hash = 0xAAAABBBBCCCCDDDDu64;

        // --- Measure Witness Construction ---
        let mut m_vars: Option<usize> = None; // Store the number of variables for MLEs.
                                              // Measure time for `HammingDistanceProof::new`.
        let (first_witness, witness_durations) = measure_many(cfg.warmup, cfg.samples, || {
            // Create a new instance. This involves data loading and MLE setup.
            let p = HammingDistanceProof::<F, G>::new(a_hash, &hash_file_path, 0).unwrap();
            // Capture the number of variables from the created proof.
            if m_vars.is_none() {
                m_vars = Some(p.d_mle.num_vars());
            }
        });
        let m_vars = m_vars.expect("MLE variable count should be determined");
        // Compute statistics for witness construction time.
        let w_stats = compute_stats(&witness_durations);

        // --- Generate Challenges ---
        // Generate challenges (alpha, gamma, lambda) and random points (r_y, z) once per size
        // to ensure that the prove and verify steps use the same random values for fair comparison.
        use rand::SeedableRng;
        let mut rng = if let Some(seed) = cfg.seed {
            // Seed RNG based on the provided seed and a constant to differentiate from witness seeding.
            rand::rngs::StdRng::seed_from_u64(seed ^ 0x55AA_1234)
        } else {
            rand::rngs::StdRng::from_entropy()
        };
        let alpha: F = F::rand(&mut rng); // Phase 1 challenge.
        let gamma: F = F::rand(&mut rng); // Phase 2 challenge.
        let lambda: F = F::rand(&mut rng); // Phase 2 challenge.
                                           // Random points for Sumcheck protocols.
        let r_y: Vec<F> = (0..m_vars).map(|_| F::rand(&mut rng)).collect();
        let z: Vec<F> = (0..m_vars).map(|_| F::rand(&mut rng)).collect();

        // --- Measure Proof Generation and Verification ---
        let (first_prove, prove_durations) = {
            // Need to re-initialize the prover for each prove measurement to include witness construction time if desired,
            // but typically we want to measure *only* the prove step assuming witness is ready.
            // Here, we measure the full `generate_full_proof` which includes witness setup implicitly if needed,
            // but the `new` call happens outside the measurement loop.

            // `measure_many` includes the call to `HammingDistanceProof::new` inside its closure.
            // For accurate `prove` time, we should ideally measure only `generate_full_proof`
            // assuming the prover instance is already constructed.
            // Let's refine this: create prover once, then measure `generate_full_proof`.

            let mut prover = HammingDistanceProof::<F, G>::new(a_hash, &hash_file_path, 0).unwrap();
            let mut last_proof; // To hold the last generated proof for verification.

            // Measure the first prove operation.
            let t0 = Instant::now();
            last_proof = prover
                .generate_full_proof(alpha, gamma, lambda, &r_y, &z)
                .unwrap();
            let first = t0.elapsed();

            // Perform warmup runs for proving.
            for _ in 0..cfg.warmup.saturating_sub(1) {
                // Re-create prover for each run to measure `new` + `generate_full_proof`.
                // If measuring *only* `generate_full_proof`, reuse `prover`.
                // For this benchmark structure, it seems `new` is part of the measured `prove` time.
                let mut p = HammingDistanceProof::<F, G>::new(a_hash, &hash_file_path, 0).unwrap();
                let _ = p
                    .generate_full_proof(alpha, gamma, lambda, &r_y, &z)
                    .unwrap();
            }

            // Collect sampled prove measurements.
            let mut out = Vec::with_capacity(cfg.samples);
            for _ in 0..cfg.samples {
                let mut p = HammingDistanceProof::<F, G>::new(a_hash, &hash_file_path, 0).unwrap();
                let t = Instant::now();
                last_proof = p
                    .generate_full_proof(alpha, gamma, lambda, &r_y, &z)
                    .unwrap();
                out.push(t.elapsed());
            }

            // --- Measure Verification ---
            // Serialize the proof once to get its size.
            let mut proof_bytes = Vec::new();
            ark_serialize::CanonicalSerialize::serialize_uncompressed(
                &last_proof, // Use the last generated proof.
                &mut proof_bytes,
            )
            .unwrap();
            let proof_size = proof_bytes.len();

            // Create a verifier instance (can reuse the prover instance conceptually).
            let verifier = HammingDistanceProof::<F, G>::new(a_hash, &hash_file_path, 0).unwrap();

            // Measure the first verification.
            let t_first_v = Instant::now();
            let ok = verifier.verify(&last_proof).unwrap();
            assert!(ok, "Verification of the first proof failed!"); // Assert validity.
            let first_v = t_first_v.elapsed();

            // Perform warmup runs for verification.
            for _ in 0..cfg.warmup.saturating_sub(1) {
                let ok = verifier.verify(&last_proof).unwrap();
                assert!(ok);
            }

            // Collect sampled verification measurements.
            let mut verify_vec = Vec::with_capacity(cfg.samples);
            for _ in 0..cfg.samples {
                let t = Instant::now();
                let ok = verifier.verify(&last_proof).unwrap();
                assert!(ok); // Assert validity of each proof.
                verify_vec.push(t.elapsed());
            }

            // Compute statistics for prove and verify times.
            let pr_stats = compute_stats(&out);
            let v_stats = compute_stats(&verify_vec);

            // Print results for the current size.
            println!(
                "[quick_bench] size={id} witness_first={:?} {}",
                first_witness, // Time for witness construction.
                fmt_stats("witness", &w_stats)
            );
            println!(
                "[quick_bench] size={id} prove_first={:?} {}",
                first, // Time for the first prove operation.
                fmt_stats("prove", &pr_stats)
            );
            println!(
                "[quick_bench] size={id} verify_first={:?} {}",
                first_v, // Time for the first verification.
                fmt_stats("verify", &v_stats)
            );
            println!(
                "[quick_bench] size={id} proof_size={} bytes m_vars={} warmup={} samples={} seed={:?}",
                proof_size, m_vars, cfg.warmup, cfg.samples, cfg.seed
            );
            // Return the time for the first prove operation and sampled prove durations.
            (first, out)
        };
        // Keep variables in scope even if unused, to prevent compiler warnings or potential future use.
        let _ = (first_prove, prove_durations);
    }
}
