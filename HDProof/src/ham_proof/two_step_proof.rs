use ark_ff::{batch_inversion, BigInteger, PrimeField};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension, Polynomial};
use ark_std::{rc::Rc, vec::Vec};
use ml_sumcheck::{data_structures::ListOfProductsOfPolynomials, MLSumcheck, Proof};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

use ark_crypto_primitives::sponge::poseidon::{PoseidonConfig, PoseidonSponge};
use ark_crypto_primitives::sponge::Absorb;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ec::AffineRepr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::collections::BTreeMap;
use poly_commit::{
    hyrax::{
        HyraxCommitment, HyraxCommitmentState, HyraxCommitterKey, HyraxPC, HyraxProof,
        HyraxVerifierKey,
    },
    LabeledCommitment, LabeledPolynomial, PolynomialCommitment,
};

use crate::ham_proof::tools::{pad_mle_to_even_vars, pad_point, poseidon_config_for_test};

// Verbose printing macro, controlled by the "VERBOSE" environment variable.
#[allow(unused_macros)]
macro_rules! vprintln {
    ($($arg:tt)*) => {
        if option_env!("VERBOSE").is_some() { println!($($arg)*); }
    };
}

/// Two-phase proof for Hamming Distance:
/// 1) Consistency phase: Prove d = f(a,B);
/// 2) Range phase: Prove d falls within the legal range [τ, N].
/// This is based on Sumcheck + LogUp.
pub struct HammingDistanceProof<F: PrimeField + Absorb, G: AffineRepr<ScalarField = F>> {
    // Core input/result data
    /// Bit vector of `a` (N bits).
    pub a_vector: Vec<F>,
    /// Matrix `B`: M rows of N-bit hashes.
    pub b_matrix: Vec<Vec<F>>,
    /// Vector `d`: Computed Hamming distances.
    pub d_vector: Vec<F>,

    // Multilinear Extensions (MLEs)
    /// MLE for `a_vector`: ã(x) over {0,1}^n.
    pub a_mle: Rc<DenseMultilinearExtension<F>>,
    /// MLE for `b_matrix`: B̃(y,x) over {0,1}^{n+m}.
    pub b_mle: Rc<DenseMultilinearExtension<F>>,
    /// MLE for `d_vector`: d̃(y) over {0,1}^m.
    pub d_mle: Rc<DenseMultilinearExtension<F>>,

    // Phase 1 (Consistency Proof) related data
    /// Randomly weighted polynomial α(y).
    pub alpha_mle: Rc<DenseMultilinearExtension<F>>,
    /// Linear parameters for efficient computation of α(r).
    pub alpha_linear_params: Option<Vec<(F, F)>>,
    /// Auxiliary polynomial g(y) = α(y)(d(y)-f(y)).
    pub g_mle: Option<Rc<DenseMultilinearExtension<F>>>,

    // Phase 2 (Range Proof) related data
    /// MLE for lookup table `T`: Contains all valid Hamming distance values [τ, N].
    pub t_mle: Rc<DenseMultilinearExtension<F>>,
    /// MLE for counts `m`: Records the frequency of each `T` value in `d_vector`.
    pub m_mle: Rc<DenseMultilinearExtension<F>>,
    /// MLE of `t_mle` lifted to the `m`-variable domain (for unified variable space with `d`, `h`).
    pub lifted_t_mle: Option<Rc<DenseMultilinearExtension<F>>>,
    /// MLE of `m_mle` lifted to the `m`-variable domain.
    pub lifted_m_mle: Option<Rc<DenseMultilinearExtension<F>>>,
    /// Auxiliary polynomial h(y) = 1/(γ+d(y)) - (N/M) * m(y_low)/(γ+t(y_low)).
    pub h_mle: Option<Rc<DenseMultilinearExtension<F>>>,
    /// Cache for `h_mle` components, containing MLEs for `1/(γ+d(y))` and `m(y_low)/(γ+t(y_low))`.
    pub h_components_cache: HComponentsCache<F>,

    // Scale parameters
    /// Number of bits in input `a`, N.
    pub n_size: usize,
    /// Number of rows in matrix `B`, M (rounded up to the nearest power of two).
    pub m_size: usize,
    /// Normalization coefficient N/M (as a Field element).
    pub n_over_m: F,

    // Hyrax PCS keys and configuration
    poseidon_config: PoseidonConfig<F>,
    ck_d: Option<HyraxCommitterKey<G>>, // For m-variable polynomials (d, h)
    vk_d: Option<HyraxVerifierKey<G>>,
    ck_m: Option<HyraxCommitterKey<G>>, // For m-variable polynomial (m)
    vk_m: Option<HyraxVerifierKey<G>>,
    ck_a: Option<HyraxCommitterKey<G>>, // For n-variable polynomial (a)
    vk_a: Option<HyraxVerifierKey<G>>,
    ck_b: Option<HyraxCommitterKey<G>>, // For n-variable polynomial (B(r, X))
    vk_b: Option<HyraxVerifierKey<G>>,
    /// Stores states needed for evaluating committed polynomials at specific points.
    commitment_states: BTreeMap<&'static str, (HyraxCommitmentState<F>, bool)>,

    // Polynomial information required for Sumcheck protocol
    /// Information about polynomial `g` (number of variables, max number of multiplicands).
    pub g_info: ml_sumcheck::data_structures::PolynomialInfo,
    /// Information about polynomial `f` (number of variables, max number of multiplicands).
    pub f_info: ml_sumcheck::data_structures::PolynomialInfo,
    /// Information about the merged range proof polynomial (number of variables, max number of multiplicands).
    pub range_info: ml_sumcheck::data_structures::PolynomialInfo,
}

/// Type alias for the cache of `h_mle` components.
type HComponentsCache<F> = Option<(
    Rc<DenseMultilinearExtension<F>>, // Left term MLE: 1/(γ+d(y))
    Rc<DenseMultilinearExtension<F>>, // Right term MLE: m(y_low)/(γ+t(y_low))
)>;

// TODO: Consider if Clone is necessary for h_mle in production, it should be committed, not recomputed.
impl<F: PrimeField + Absorb, G: AffineRepr<ScalarField = F>> Clone for HammingDistanceProof<F, G> {
    fn clone(&self) -> Self {
        HammingDistanceProof {
            a_vector: self.a_vector.clone(),
            b_matrix: self.b_matrix.clone(),
            d_vector: self.d_vector.clone(),
            a_mle: self.a_mle.clone(),
            b_mle: self.b_mle.clone(),
            d_mle: self.d_mle.clone(),
            alpha_mle: self.alpha_mle.clone(),
            alpha_linear_params: self.alpha_linear_params.clone(),
            g_mle: self.g_mle.clone(),
            t_mle: self.t_mle.clone(),
            m_mle: self.m_mle.clone(),
            lifted_t_mle: self.lifted_t_mle.clone(),
            lifted_m_mle: self.lifted_m_mle.clone(),
            h_mle: self.h_mle.clone(),
            h_components_cache: self
                .h_components_cache
                .as_ref()
                .map(|(left, right)| (left.clone(), right.clone())),
            n_size: self.n_size,
            m_size: self.m_size,
            n_over_m: self.n_over_m,
            poseidon_config: self.poseidon_config.clone(),
            ck_d: None, // Keys are typically not cloned in this context
            vk_d: None,
            ck_m: None,
            vk_m: None,
            ck_a: None,
            vk_a: None,
            ck_b: None,
            vk_b: None,
            commitment_states: BTreeMap::new(), // Commitment states are specific to an instance/prover
            g_info: self.g_info.clone(),
            f_info: self.f_info.clone(),
            range_info: self.range_info.clone(),
        }
    }
}

/// Result of the two-phase proof.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct FullProof<F: PrimeField, G: AffineRepr<ScalarField = F>> {
    // Phase 1 (Consistency) proof data
    /// Sumcheck proof for g(y), proving Σ g(y) = 0.
    pub g_proof: Proof<F>,
    /// Sumcheck proof for f(y) at challenge point r, proving the value f(r).
    pub f_proof: Proof<F>,
    /// Random challenge point r (y-domain) generated by the Sumcheck protocol.
    pub r_vec: Vec<F>,
    /// Computed value of f(y) at challenge point r, f(r).
    pub f_r_val: F,
    /// Random challenge value α for Phase 1.
    pub alpha: F,

    // Phase 2 (Range) proof data
    /// Sumcheck proof for the merged range polynomial Q(y), proving Σ Q(y) = 0.
    pub range_proof: Proof<F>,
    /// Random challenge value γ for Phase 2.
    pub gamma: F,
    /// Random challenge value λ for Phase 2.
    pub lambda: F,
    /// Random challenge point z generated by the Sumcheck protocol.
    pub z_vec: Vec<F>,

    // Hyrax Commitments (optional)
    /// Commitment to d̃(y).
    pub d_comm: Option<HyraxCommitment<G>>,
    /// Commitment to m̃(y_low).
    pub m_comm: Option<HyraxCommitment<G>>,
    /// Commitment to ã(x).
    pub a_comm: Option<HyraxCommitment<G>>,
    /// Commitment to B̃(r_y, x) (MLE obtained after partial evaluation of B over y).
    pub b_comm: Option<HyraxCommitment<G>>,
    /// Commitment to h̃(y).
    pub h_comm: Option<HyraxCommitment<G>>,

    // Evaluated values of committed polynomials at challenge points
    /// Evaluation of d̃(y) at point r, d̃(r_y).
    pub d_at_r: Option<F>,
    /// Evaluation of ã(x) at point r_x, ã(r_x).
    pub a_at_rx: Option<F>,
    /// Evaluation of B̃(r_y, x) at point r_x, B̃(r_y, r_x).
    pub b_at_rx: Option<F>,
    /// Evaluation of d̃(y) at point r_q, d̃(r_q).
    pub d_at_rq: Option<F>,
    /// Evaluation of m̃(y_low) at point r_q, m̃(r_q).
    pub m_at_rq: Option<F>,
    /// Evaluation of h̃(y) at point r_q, h̃(r_q).
    pub h_at_rq: Option<F>,

    // Hyrax Opening Proofs
    /// Opening proof for d̃(y) at point r.
    pub proof_open_d_at_r: Option<Vec<HyraxProof<G>>>,
    /// Opening proof for ã(x) at point r_x (may be batched with b_at_rx).
    pub proof_open_a_at_rx: Option<Vec<HyraxProof<G>>>,
    /// Opening proof for B̃(r_y, x) at point r_x (may be batched with a_at_rx).
    pub proof_open_b_at_rx: Option<Vec<HyraxProof<G>>>,
    /// Batched opening proof for d̃(y) and h̃(y) at point r_q.
    pub proof_open_dh_at_rq: Option<Vec<HyraxProof<G>>>,
    /// Opening proof for m̃(y_low) at point r_q (may be batched with d, h).
    pub proof_open_m_at_rq: Option<Vec<HyraxProof<G>>>,
}

impl<F: PrimeField + Absorb, G: AffineRepr<ScalarField = F>> HammingDistanceProof<F, G> {
    /// Creates a new instance: requires a_hash (64-bit), hash_file_path (hex hashes), and range_threshold τ.
    pub fn new(
        a_hash: u64,
        hash_file_path: &str,
        range_threshold: u8,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // --- Data Loading and Preprocessing ---
        let b_hashes = Self::read_hash_file(hash_file_path)?;
        let a_vector_u8 = Self::hash_to_bit_vector(a_hash);

        let b_matrix_u8: Vec<Vec<u8>> = b_hashes
            .par_iter()
            .map(|&h| Self::hash_to_bit_vector(h))
            .collect();

        let n = Self::log2_ceil(a_vector_u8.len()); // Number of variables for 'a'
        let m = Self::log2_ceil(b_matrix_u8.len()); // Number of variables for 'd' and 'B's first part
        let required_len = 1 << m; // Required length for d_vector MLE evaluations

        // Compute Hamming distances and pad
        let mut d_vector_u8 = Self::compute_hamming_distances(&a_vector_u8, &b_matrix_u8);
        d_vector_u8.resize(required_len, 0); // Pad with zeros

        // Convert byte vectors to Field elements
        let a_vector: Vec<F> = a_vector_u8.par_iter().map(|&x| F::from(x)).collect();
        let b_matrix: Vec<Vec<F>> = b_matrix_u8
            .par_iter()
            .map(|row| row.par_iter().map(|&x| F::from(x)).collect())
            .collect();
        let d_vector: Vec<F> = d_vector_u8.par_iter().map(|&x| F::from(x as u64)).collect();

        // --- Construct Core MLEs ---
        let a_mle = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            n,
            a_vector.clone(),
        ));
        // Flatten b_matrix for MLE construction (interleaving y and x variables)
        let b_flat = Self::flatten_b_matrix(&b_matrix_u8, n, m);
        let b_mle = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            n + m, // Total variables: m for y, n for x
            b_flat.par_iter().map(|&x| F::from(x)).collect(),
        ));
        let d_mle = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            m, // 'd' depends only on 'y' (m variables)
            d_vector.clone(),
        ));
        // Initialize alpha_mle with zeros, will be set properly in `set_alpha`
        let alpha_mle = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            m,
            vec![F::zero(); required_len],
        ));

        // --- Phase 2: Build Lookup Table MLEs ---
        let n_bits = a_vector_u8.len() as u8;
        let (t_mle, m_mle) = Self::build_lookup_mles(&d_vector_u8, range_threshold, n_bits, n)?;

        // Setup Hyrax PCS keys
        let mut rng = ark_std::test_rng();
        let poseidon_config = poseidon_config_for_test::<F>();

        // Determine padded variable counts for PCS setup
        let m_vars = m;
        let n_vars = n;
        let m_vars_padded = if m_vars % 2 != 0 { m_vars + 1 } else { m_vars };
        let n_vars_padded = if n_vars % 2 != 0 { n_vars + 1 } else { n_vars };

        // Keys for m-variable polynomials (d, h)
        let pcs_pp_d =
            HyraxPC::<G, DenseMultilinearExtension<F>>::setup(1, Some(m_vars_padded), &mut rng)?;
        let (ck_d, vk_d) = HyraxPC::<G, DenseMultilinearExtension<F>>::trim(&pcs_pp_d, 1, 1, None)?;

        // Keys for m-variable polynomial (m). Reusing keys for simplicity.
        let (ck_m, vk_m) = (ck_d.clone(), vk_d.clone());

        // Keys for n-variable polynomial (a)
        let pcs_pp_a =
            HyraxPC::<G, DenseMultilinearExtension<F>>::setup(1, Some(n_vars_padded), &mut rng)?;
        let (ck_a, vk_a) = HyraxPC::<G, DenseMultilinearExtension<F>>::trim(&pcs_pp_a, 1, 1, None)?;

        // Keys for n-variable polynomial (B(r, X)). Reusing keys for simplicity.
        let (ck_b, vk_b) = (ck_a.clone(), vk_a.clone());

        let n_size = a_vector_u8.len();
        let m_size = b_matrix_u8.len().next_power_of_two(); // M rounded up
        let n_over_m = F::from(n_size as u64) * F::from(m_size as u64).inverse().unwrap(); // N/M as Field element

        Ok(HammingDistanceProof {
            a_vector,
            b_matrix,
            d_vector,
            a_mle,
            b_mle,
            d_mle,
            alpha_mle,
            alpha_linear_params: None, // Initialized later by set_alpha
            g_mle: None,               // Constructed in prove_phase1
            t_mle: Rc::new(t_mle),
            m_mle: Rc::new(m_mle),
            lifted_t_mle: None, // Constructed in lift_lookup_to_m_domain
            lifted_m_mle: None, // Constructed in lift_lookup_to_m_domain
            h_mle: None,        // Constructed in construct_h_mle
            h_components_cache: None,
            n_size,
            m_size,
            n_over_m,
            poseidon_config,
            ck_d: Some(ck_d),
            vk_d: Some(vk_d),
            ck_m: Some(ck_m),
            vk_m: Some(vk_m),
            ck_a: Some(ck_a),
            vk_a: Some(vk_a),
            ck_b: Some(ck_b),
            vk_b: Some(vk_b),
            commitment_states: BTreeMap::new(), // Initialized empty
            g_info: ml_sumcheck::data_structures::PolynomialInfo {
                num_variables: m,
                max_multiplicands: 1, // g(y) is a single MLE
            },
            f_info: ml_sumcheck::data_structures::PolynomialInfo {
                num_variables: n,
                max_multiplicands: 2, // f = a + B - 2*a*B involves products
            },
            range_info: ml_sumcheck::data_structures::PolynomialInfo {
                num_variables: m,
                max_multiplicands: 4, // Q(y) can involve products of eq, h, d, t
            },
        })
    }

    /// Constructs the auxiliary polynomial g(y) = α(y)(d(y)-f(y)).
    /// If d=f, then g≡0.
    /// f(y) = Σ_x (a(x) + B(y,x) - 2 * a(x) * B(y,x)).
    pub fn construct_witness_g_polynomial(&mut self, alpha: F) -> Result<(), crate::Error> {
        let m = self.d_mle.num_vars();
        let total_y = self.b_matrix.len(); // Number of actual B rows provided

        self.set_alpha(alpha); // Sets self.alpha_mle and self.alpha_linear_params
        let alpha_values = &self.alpha_mle.evaluations;

        let a_vals = &self.a_mle.evaluations;
        let a_sum: F = a_vals.par_iter().cloned().sum(); // Sum of all a(x) values
        let d_vals = &self.d_mle.evaluations;

        // Clone necessary data to avoid borrowing issues in parallel iteration
        let b_matrix_clone = self.b_matrix.clone();
        let alpha_vals_clone = alpha_values.clone();
        let a_vals_clone = a_vals.clone();
        let d_vals_clone = d_vals.clone();

        // Calculate g(y) for all 2^m possible y values
        let g_values: Vec<F> = (0..(1 << m))
            .into_par_iter()
            .map(|y_idx| {
                // Calculate f(y) for the current y_idx
                let f_y = if y_idx < total_y {
                    // If y_idx corresponds to an actual row in B
                    let b_row = &b_matrix_clone[y_idx];
                    let b_sum: F = b_row.par_iter().cloned().sum(); // Sum of b(y, x) for fixed y
                    let ab_sum: F = a_vals_clone
                        .par_iter()
                        .zip(b_row.par_iter())
                        .map(|(a_bit, b_bit)| *a_bit * *b_bit) // Compute a(x) * b(y, x)
                        .sum(); // Sum over x
                    a_sum + b_sum - F::from(2u64) * ab_sum // f(y) = sum(a) + sum(B) - 2 * sum(aB)
                } else {
                    // If y_idx is beyond the provided B rows, assume B(y,x) = 0 for this y.
                    // Then f(y) = Σ_x a(x) = a_sum.
                    a_sum
                };
                // Compute g(y) = α(y) * (d(y) - f(y))
                alpha_vals_clone[y_idx] * (d_vals_clone[y_idx] - f_y)
            })
            .collect();

        self.g_mle = Some(Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            m, g_values,
        )));
        Ok(())
    }

    /// Proves that Σ g(y) = 0 using Sumcheck.
    fn prove_g_sum_zero(&self) -> Result<Proof<F>, crate::Error> {
        let g_mle = self.g_mle.as_ref().ok_or(crate::Error::OtherError(
            "g_mle not constructed".to_string(),
        ))?;
        let mut polynomial = ListOfProductsOfPolynomials::new(g_mle.num_vars());
        // g(y) is a single MLE, so add it as a product of one polynomial.
        polynomial.add_product(vec![g_mle.clone()], F::one());
        MLSumcheck::prove(&polynomial).map_err(Into::into)
    }

    /// Proves the evaluation of f(y) at a random point r.
    /// Computes f(r) = Σ_x (ã(x) + B̃(r,x) - 2ã(x)B̃(r,x)) using Sumcheck.
    fn prove_f_evaluation_at_r(&self, r: &[F]) -> Result<(F, Proof<F>), crate::Error> {
        let n = self.a_mle.num_vars();
        // Partially evaluate B̃(y,x) at y=r to get an MLE over x: B̃(r,x).
        let b_r_values = Self::partial_evaluate(&self.b_mle, r);
        let b_r_mle = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            n, b_r_values,
        ));

        // Construct the polynomial for Sumcheck: a(x) + B̃(r,x) - 2*a(x)*B̃(r,x)
        let mut polynomial = ListOfProductsOfPolynomials::new(n);
        polynomial.add_product(vec![self.a_mle.clone()], F::one()); // Term ã(x)
        polynomial.add_product(vec![b_r_mle.clone()], F::one()); // Term B̃(r,x)
        polynomial.add_product(vec![self.a_mle.clone(), b_r_mle], -F::from(2u64)); // Term -2*ã(x)*B̃(r,x)

        // Generate the Sumcheck proof and extract the sum (which is f(r)).
        let proof = MLSumcheck::prove(&polynomial)?;
        let f_r = MLSumcheck::extract_sum(&proof);
        Ok((f_r, proof))
    }

    /// Phase 2: Range Proof. LogUp core idea: Σ 1/(γ+d) = Σ m/(γ+t).
    /// Constructs h = 1/(γ+d) - m/(γ+t).
    /// 1) Proves Σ h(y) = 0.
    /// 2) Verifies the construction using a random point z (via Q(y) = h(y) + λ*eq(y,z)*P(y)).
    pub fn prove_range_validity(
        &mut self,
        gamma: F,
        lambda: F,
        z: &[F],
    ) -> Result<Proof<F>, crate::Error> {
        let m = self.d_mle.num_vars();
        // Ensure h_mle is constructed if not already.
        if self.h_mle.is_none() {
            self.construct_h_mle(gamma)?;
        }
        // Ensure lookup tables are lifted to the correct domain if necessary.
        if self.lifted_t_mle.is_none() || self.lifted_m_mle.is_none() {
            self.lift_lookup_to_m_domain();
        }
        let h_mle = self.h_mle.as_ref().unwrap();
        let t_lift = self.lifted_t_mle.as_ref().unwrap(); // t lifted to m vars
        let m_lift = self.lifted_m_mle.as_ref().unwrap(); // m lifted to m vars
        let d_mle = &self.d_mle;
        let eq_z = Self::compute_eq_mle(z)?; // Kronecker delta MLE eq(y,z)
        let m_f = F::from(self.m_size as u64); // M as Field element
        let n_f = F::from(self.n_size as u64); // N as Field element
        let gamma_sq = gamma * gamma;

        // Construct the merged polynomial Q(y) = h(y) + λ * eq(y,z) * P(y)
        // where P(y) = M*(γ+t_low) - N*m_low*(γ+d) - M*γ*(γ+d)*(h+t_low) - M*(γ+d)*h*t_low
        // The goal is to prove Σ Q(y) = 0
        let mut poly = ListOfProductsOfPolynomials::new(m);

        // Term 1: h(y)
        poly.add_product(vec![h_mle.clone()], F::one());

        // Terms involving λ * eq(y,z) * P(y)
        // P(y) = M*(γ+t_low) - N*m_low*(γ+d) - M*γ*h*(γ+t_low) - M*(γ+d)*h*t_low  (corrected from original comment)
        // The below terms correspond to the expansion of the equation check.
        // This requires careful reconstruction of the polynomial based on the prover's logic.

        // Term: λ * eq(y,z) * M * γ
        poly.add_product(vec![eq_z.clone()], lambda * m_f * gamma);
        // Term: λ * eq(y,z) * M * t_low
        poly.add_product(vec![eq_z.clone(), t_lift.clone()], lambda * m_f);
        // Term: λ * eq(y,z) * (-N * γ * m_low)
        poly.add_product(vec![eq_z.clone(), m_lift.clone()], -lambda * n_f * gamma);
        // Term: λ * eq(y,z) * (-N * m_low * d)
        poly.add_product(
            vec![eq_z.clone(), m_lift.clone(), d_mle.clone()],
            -lambda * n_f,
        );
        // Term: λ * eq(y,z) * (-M * γ^2 * h)
        poly.add_product(vec![eq_z.clone(), h_mle.clone()], -lambda * m_f * gamma_sq);
        // Term: λ * eq(y,z) * (-M * γ * h * t_low)
        poly.add_product(
            vec![eq_z.clone(), h_mle.clone(), t_lift.clone()],
            -lambda * m_f * gamma,
        );
        // Term: λ * eq(y,z) * (-M * γ * h * d)
        poly.add_product(
            vec![eq_z.clone(), h_mle.clone(), d_mle.clone()],
            -lambda * m_f * gamma,
        );
        // Term: λ * eq(y,z) * (-M * h * d * t_low)
        poly.add_product(
            vec![eq_z.clone(), h_mle.clone(), d_mle.clone(), t_lift.clone()],
            -lambda * m_f,
        );

        // Generate the Sumcheck proof for Q(y).
        MLSumcheck::prove(&poly).map_err(Into::into)
    }

    /// Lifts the `t_mle` and `m_mle` (which are defined over `n` variables)
    /// to the `m`-variable domain used by `d_mle` and `h_mle`.
    /// This is done by replicating the evaluation vector.
    fn lift_lookup_to_m_domain(&mut self) {
        let n_vars = self.t_mle.num_vars();
        let m_vars = self.d_mle.num_vars();
        // If already lifted or if variable counts match, return.
        if self.lifted_t_mle.is_some() {
            return;
        }
        if n_vars == m_vars {
            self.lifted_t_mle = Some(self.t_mle.clone());
            self.lifted_m_mle = Some(self.m_mle.clone());
            return;
        }
        // Calculate repetition factor.
        let repeat = 1 << (m_vars - n_vars);
        let base_len = 1 << n_vars;
        let mut t_ext = Vec::with_capacity(base_len * repeat);
        let mut m_ext = Vec::with_capacity(base_len * repeat);
        // Replicate the evaluations `repeat` times.
        for _ in 0..repeat {
            t_ext.extend_from_slice(&self.t_mle.evaluations);
        }
        for _ in 0..repeat {
            m_ext.extend_from_slice(&self.m_mle.evaluations);
        }
        // Create new MLEs with `m_vars` variables.
        self.lifted_t_mle = Some(Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            m_vars, t_ext,
        )));
        self.lifted_m_mle = Some(Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            m_vars, m_ext,
        )));
    }

    /// Constructs the auxiliary polynomial h(y) = 1/(γ+d(y)) - (N/M) * m(y_low)/(γ+t(y_low)).
    /// Also caches the MLEs for its two main components: 1/(γ+d(y)) and m/(γ+t).
    fn construct_h_mle(&mut self, gamma: F) -> Result<(), crate::Error> {
        let m_vars = self.d_mle.num_vars();
        let n_vars = self.t_mle.num_vars();
        assert!(
            m_vars >= n_vars,
            "m_vars must be >= n_vars for lookup lifting"
        );
        let total = 1 << m_vars;
        let mask = (1usize << n_vars) - 1; // Mask to extract the lower n_vars bits of y
        let d_evals = &self.d_mle.evaluations;
        let t_evals = &self.t_mle.evaluations;
        let m_small = &self.m_mle.evaluations; // Counts (m) are defined over n_vars
        let n_over_m = self.n_over_m; // Precomputed N/M

        // Compute denominators: (γ + d(y)) and (γ + t(y_low))
        let mut denom_d: Vec<F> = (0..total)
            .into_par_iter()
            .map(|i| gamma + d_evals[i])
            .collect();
        let mut denom_t: Vec<F> = (0..total)
            .into_par_iter()
            .map(|i| {
                let low = i & mask; // Extract lower n_vars bits for t lookup
                gamma + t_evals[low]
            })
            .collect();

        // Perform batch inversion for efficiency
        batch_inversion(&mut denom_d);
        batch_inversion(&mut denom_t);

        // Compute h(y) and its components
        let mut h = Vec::with_capacity(total);
        let mut left = Vec::with_capacity(total); // Stores 1/(γ+d(y))
        let mut right = Vec::with_capacity(total); // Stores (N/M) * m(y_low)/(γ+t(y_low))

        for i in 0..total {
            let low = i & mask; // Index for t and m (lower n_vars bits)
            let inv_d = denom_d[i]; // 1 / (γ + d(y))
            let inv_t = denom_t[i]; // 1 / (γ + t(y_low))
            let term_right = n_over_m * m_small[low] * inv_t; // (N/M) * m(y_low) / (γ + t(y_low))

            left.push(inv_d);
            right.push(term_right);
            h.push(inv_d - term_right); // h(y) = left - right
        }

        // Create MLEs for h(y) and its components
        self.h_mle = Some(Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            m_vars, h,
        )));
        self.h_components_cache = Some((
            Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                m_vars, left,
            )),
            Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                m_vars, right,
            )),
        ));
        Ok(())
    }

    /// Verifies the proof.
    /// Phase 1: Checks Σg=0 and consistency of f(r) and g(r).
    /// Phase 2: Checks ΣQ=0.
    /// Optionally verifies Hyrax openings if commitments are provided.
    pub fn verify(&self, proof: &FullProof<F, G>) -> Result<bool, crate::Error> {
        // --- Phase 1 Verification ---
        // 1. Verify that the sum of g(y) is zero. Extract claimed sum for verification.
        let g_sum_claimed = MLSumcheck::extract_sum(&proof.g_proof);
        if g_sum_claimed != F::zero() {
            vprintln!("Phase 1 Verify FAILED: g(y) sum is not zero (is {g_sum_claimed})");
            return Ok(false);
        }

        // 2. Verify the Sumcheck proof for f(r) and check consistency.
        //    Extract the claimed sum (f(r)) from the f_proof.
        let f_sum_claimed = MLSumcheck::extract_sum(&proof.f_proof);
        // Check if the claimed sum matches the claimed evaluation f_r_val.
        if f_sum_claimed != proof.f_r_val {
            vprintln!("Phase 1 Verify FAILED: f(r) sumcheck mismatch");
            return Ok(false);
        }

        // Independently verify the Sumcheck proofs for g and f.
        let g_info_local = self.g_info.clone();
        let f_info_local = self.f_info.clone();
        let (g_res, f_res) = rayon::join(
            || MLSumcheck::verify(&g_info_local, g_sum_claimed, &proof.g_proof),
            || MLSumcheck::verify(&f_info_local, proof.f_r_val, &proof.f_proof),
        );
        let g_subclaim = g_res?; // Result of verifying g's Sumcheck proof
        let f_subclaim = f_res?; // Result of verifying f's Sumcheck proof

        // Extract the evaluated point and value from the Sumcheck results.
        let verified_g_r = g_subclaim.expected_evaluation; // g(r) as computed by the verifier
        let r_x_verified = f_subclaim.point.clone(); // r_x used in f's Sumcheck

        // 3. Perform consistency check: g(r) = α(r) * (d(r) - f(r))
        // Calculate α(r) using the challenge alpha and the random point r_vec.
        let alpha_r = Self::alpha_value_at_from_challenge(proof.alpha, &proof.r_vec);
        // Get d(r). Prefer using the opened value from the proof if available, otherwise evaluate locally (for testing).
        let d_r = proof
            .d_at_r
            .unwrap_or_else(|| self.d_mle.evaluate(&proof.r_vec)); // Evaluate d(r) if not provided
                                                                   // Calculate the expected g(r) based on the consistency equation.
        let expected_g_r = alpha_r * (d_r - proof.f_r_val);

        // Compare the verifier's g(r) with the expected g(r).
        if verified_g_r != expected_g_r {
            vprintln!("Phase 1 Verify FAILED: Consistency check failed");
            vprintln!("  Verified g(r): {verified_g_r}");
            vprintln!("  Expected g(r): {expected_g_r}");
            return Ok(false);
        }
        vprintln!("Phase 1 Verify SUCCESS");

        // --- Phase 2 Verification ---
        // Verify the Sumcheck proof for the merged range polynomial Q(y).
        let range_sum = MLSumcheck::extract_sum(&proof.range_proof);
        if range_sum != F::zero() {
            vprintln!("Phase 2 Verify FAILED: range polynomial sum != 0 (is {range_sum})");
            return Ok(false);
        }
        // Verify the Sumcheck proof itself and get the challenge point r_q.
        let range_sub = MLSumcheck::verify(&self.range_info, range_sum, &proof.range_proof)?;
        let r_q = range_sub.point.clone(); // Challenge point for Phase 2 openings
        vprintln!("Phase 2 Verify SUCCESS");

        // --- Hyrax Checks (Optional): Verify openings if commitments are provided ---
        if let (
            Some(vk_d),
            Some(vk_m),
            Some(vk_a),
            Some(vk_b),
            Some(d_comm),
            Some(m_comm),
            Some(a_comm),
            Some(b_comm),
        ) = (
            self.vk_d.clone(),
            self.vk_m.clone(),
            self.vk_a.clone(),
            self.vk_b.clone(),
            proof.d_comm.clone(),
            proof.m_comm.clone(),
            proof.a_comm.clone(),
            proof.b_comm.clone(),
        ) {
            // Initialize Sponge for Hyrax checks
            let poseidon_config = self.poseidon_config.clone();
            // Pad points if necessary for Hyrax check function (depends on polynomial structure)
            let d_vars_odd = self.d_mle.num_vars() % 2 != 0;
            let a_vars_odd = self.a_mle.num_vars() % 2 != 0;
            let m_vars_odd = self.m_mle.num_vars() % 2 != 0;
            let r_y_padded = pad_point(&proof.r_vec, d_vars_odd); // Point r for d evaluation
            let r_x_padded = pad_point(&r_x_verified, a_vars_odd); // Point r_x for a and b evaluations
            let rq_padded_d = pad_point(&r_q, d_vars_odd); // Point r_q for d and h evaluations
            let rq_padded_m = pad_point(&r_q, m_vars_odd); // Point r_q for m evaluation

            // Task 1: Verify openings at r (for d) and r_x (for a and b).
            let rx_task = || -> Result<bool, crate::Error> {
                let mut ok = true;
                let sponge = PoseidonSponge::<F>::new(&poseidon_config);

                // Verify opening of d̃(y) at r_y
                if let (Some(proofs), Some(d_at_r)) = (&proof.proof_open_d_at_r, proof.d_at_r) {
                    let d_lc = LabeledCommitment::new("d".to_string(), d_comm.clone(), None);
                    let mut sp = sponge.clone();
                    ok &= HyraxPC::<G, DenseMultilinearExtension<F>>::check(
                        &vk_d, // Verifier key for m-variable polynomials
                        &[d_lc],
                        &r_y_padded,
                        [d_at_r],
                        proofs,
                        &mut sp,
                        None, // No auxiliary data needed for this check
                    )?;
                }

                // Verify openings of ã(x) and B̃(r_y, x) at r_x.
                // These might be batched in a single proof.
                if let (Some(proofs_a), Some(proofs_b), Some(a_at_rx), Some(b_at_rx)) = (
                    &proof.proof_open_a_at_rx,
                    &proof.proof_open_b_at_rx,
                    proof.a_at_rx,
                    proof.b_at_rx,
                ) {
                    let a_lc = LabeledCommitment::new("a".to_string(), a_comm.clone(), None);
                    let b_lc = LabeledCommitment::new("b".to_string(), b_comm.clone(), None);
                    let mut sp = sponge.clone();
                    // Attempt batched verification first.
                    let batch_ok = HyraxPC::<G, DenseMultilinearExtension<F>>::check(
                        &vk_a,                         // Verifier key for n-variable polynomials
                        &[a_lc.clone(), b_lc.clone()], // Both polynomials committed
                        &r_x_padded,
                        [a_at_rx, b_at_rx], // Corresponding evaluations
                        proofs_a,           // Proofs from the batch
                        &mut sp,
                        None,
                    )?;
                    // If batched verification fails, try individual verifications.
                    if !batch_ok {
                        let mut sp1 = sponge.clone();
                        ok &= HyraxPC::<G, DenseMultilinearExtension<F>>::check(
                            &vk_a,
                            &[a_lc],
                            &r_x_padded,
                            [a_at_rx],
                            proofs_a,
                            &mut sp1,
                            None,
                        )?;
                        let mut sp2 = sponge.clone();
                        ok &= HyraxPC::<G, DenseMultilinearExtension<F>>::check(
                            &vk_b, // Verifier key for B(r, X)
                            &[b_lc],
                            &r_x_padded,
                            [b_at_rx],
                            proofs_b,
                            &mut sp2,
                            None,
                        )?;
                    }
                } else {
                    // Handle cases where only one of a or b is opened.
                    if let (Some(proofs), Some(a_at_rx)) =
                        (&proof.proof_open_a_at_rx, proof.a_at_rx)
                    {
                        let a_lc = LabeledCommitment::new("a".to_string(), a_comm.clone(), None);
                        let mut sp = sponge.clone();
                        ok &= HyraxPC::<G, DenseMultilinearExtension<F>>::check(
                            &vk_a,
                            &[a_lc],
                            &r_x_padded,
                            [a_at_rx],
                            proofs,
                            &mut sp,
                            None,
                        )?;
                    }
                    if let (Some(proofs), Some(b_at_rx)) =
                        (&proof.proof_open_b_at_rx, proof.b_at_rx)
                    {
                        let b_lc = LabeledCommitment::new("b".to_string(), b_comm.clone(), None);
                        let mut sp = sponge.clone();
                        ok &= HyraxPC::<G, DenseMultilinearExtension<F>>::check(
                            &vk_b,
                            &[b_lc],
                            &r_x_padded,
                            [b_at_rx],
                            proofs,
                            &mut sp,
                            None,
                        )?;
                    }
                }
                Ok(ok)
            };

            // Task 2: Verify openings at r_q (for d, h, and m).
            let rq_task = || -> Result<bool, crate::Error> {
                let mut ok = true;
                let sponge = PoseidonSponge::<F>::new(&poseidon_config);

                // Verify openings of d̃(y) and h̃(y) at r_q (batched).
                if let (Some(proofs_dh), Some(d_at_rq), Some(h_at_rq)) =
                    (&proof.proof_open_dh_at_rq, proof.d_at_rq, proof.h_at_rq)
                {
                    if let Some(h_comm) = &proof.h_comm {
                        let d_lc = LabeledCommitment::new("d".to_string(), d_comm.clone(), None);
                        let h_lc = LabeledCommitment::new("h".to_string(), h_comm.clone(), None);
                        let mut sp = sponge.clone();
                        ok &= HyraxPC::<G, DenseMultilinearExtension<F>>::check(
                            &vk_d, // Verifier key for m-variable polynomials
                            &[d_lc, h_lc],
                            &rq_padded_d,
                            [d_at_rq, h_at_rq],
                            proofs_dh,
                            &mut sp,
                            None,
                        )?;
                    }
                }

                // Verify opening of m̃(y_low) at r_q.
                if let (Some(proofs_m), Some(m_at_rq)) = (&proof.proof_open_m_at_rq, proof.m_at_rq)
                {
                    let m_lc = LabeledCommitment::new("m".to_string(), m_comm.clone(), None);
                    let mut sp = sponge.clone();
                    ok &= HyraxPC::<G, DenseMultilinearExtension<F>>::check(
                        &vk_m, // Verifier key for m-variable polynomials
                        &[m_lc],
                        &rq_padded_m,
                        [m_at_rq],
                        proofs_m,
                        &mut sp,
                        None,
                    )?;
                }
                Ok(ok)
            };

            // Execute both tasks in parallel.
            let (rx_res, rq_res) = rayon::join(rx_task, rq_task);
            // Return false if any Hyrax check fails.
            if !(rx_res? && rq_res?) {
                return Ok(false);
            }
        }
        // If all checks pass, the proof is valid.
        Ok(true)
    }

    /// Builds the lookup tables `T` (valid distances) and `m` (counts) as MLEs.
    /// Pads unused table entries with sentinel values or zeros.
    fn build_lookup_mles(
        d_vector: &[u8],
        threshold: u8,
        n_bits: u8,
        m_vars: usize, // Number of variables for the lookup table MLE
    ) -> Result<(DenseMultilinearExtension<F>, DenseMultilinearExtension<F>), crate::Error> {
        let table_len = 1 << m_vars; // Size of the lookup table (2^m_vars)
                                     // Number of valid distance values in the range [threshold, n_bits].
        let valid_range_size = (n_bits - threshold + 1) as usize;

        // Choose a sentinel value that is guaranteed not to be in the valid range.
        let sentinel_value = if threshold > 0 {
            threshold - 1 // Use a value just below the threshold
        } else {
            n_bits + 1 // Use a value just above the max possible distance (64)
        };

        // 1. Construct the T vector (valid distances).
        let mut t_vec = Vec::with_capacity(table_len);
        for i in 0..table_len {
            let table_val = if i < valid_range_size {
                // For indices within the valid range, use the actual distance value.
                threshold + i as u8
            } else {
                // For indices outside the valid range, use the sentinel value.
                sentinel_value
            };
            t_vec.push(F::from(table_val as u64));
        }

        // 2. Count occurrences of each distance in d_vector.
        let mut m_counts = vec![0u64; table_len]; // Initialize counts to zero
        let mut invalid_count = 0u64; // Count of distances outside the valid range

        for &dist in d_vector.iter() {
            if dist >= threshold && dist <= n_bits {
                // If the distance is within the valid range [threshold, n_bits].
                let table_idx = (dist - threshold) as usize; // Map distance to table index
                if table_idx < valid_range_size {
                    m_counts[table_idx] += 1; // Increment count for this valid distance
                }
            } else {
                // If the distance is outside the valid range.
                invalid_count += 1;
            }
        }

        // 3. Handle invalid counts: aggregate them into the first available sentinel slot if needed.
        if invalid_count > 0 && valid_range_size < table_len {
            // Place the total count of invalid distances in the first sentinel slot.
            m_counts[valid_range_size] = invalid_count;
            // Ensure the corresponding T value is the sentinel value.
            t_vec[valid_range_size] = F::from(sentinel_value as u64);
        }

        // 4. Convert counts to Field elements for the m vector.
        let m_vec: Vec<F> = m_counts.into_iter().map(|count| F::from(count)).collect();

        // 5. Verify statistical correctness: total counts should match d_vector length.
        let total_counted: u64 = m_vec
            .iter()
            .enumerate()
            .map(|(i, &count_f)| {
                // Safely extract u64 count from Field element.
                let count_bytes = count_f.into_bigint().to_bytes_le();
                let count = if count_bytes.len() >= 8 {
                    // If enough bytes, reconstruct u64 directly.
                    u64::from_le_bytes([
                        count_bytes[0],
                        count_bytes[1],
                        count_bytes[2],
                        count_bytes[3],
                        count_bytes[4],
                        count_bytes[5],
                        count_bytes[6],
                        count_bytes[7],
                    ])
                } else {
                    // Otherwise, reconstruct from available bytes (for smaller numbers).
                    let mut bytes = [0u8; 8];
                    bytes[..count_bytes.len()].copy_from_slice(&count_bytes);
                    u64::from_le_bytes(bytes)
                };

                // Verbose logging for counts (limited to avoid excessive output).
                if i < 10 || count > 0 {
                    vprintln!(
                        "    Position {}: T[{}] = {}, m[{}] = {}",
                        i,
                        i,
                        if i < t_vec.len() {
                            format!("{}", t_vec[i])
                        } else {
                            "N/A".to_string()
                        },
                        i,
                        count
                    );
                }
                count
            })
            .sum();

        let expected_total = d_vector.len() as u64;

        vprintln!("  Total counts: {total_counted}, Expected total: {expected_total}");
        vprintln!("  Number of invalid distances: {invalid_count}");

        // Error if the total counted elements do not match the original d_vector size.
        if total_counted != expected_total {
            return Err(crate::Error::OtherError(format!(
                "Count mismatch: counted {} elements, but d_vector has {} elements",
                total_counted, expected_total
            )));
        }

        // 6. Basic check for LogUp constraint feasibility: ensure all d_vector values correspond to valid T entries.
        //    This is a sanity check, not strictly part of the Sumcheck protocol itself.
        let mut _lookup_verification_passed = true;
        for &dist in d_vector.iter().take(100) {
            // Check only first 100 for verbosity
            let mut found = false;
            // Check if 'dist' exists in the valid part of T and has a non-zero count in m.
            for (i, &t_val_f) in t_vec.iter().enumerate().take(valid_range_size + 1) {
                let t_val = t_val_f.into_bigint().to_bytes_le();
                let t_val_u8 = if !t_val.is_empty() { t_val[0] } else { 0 }; // Extract the byte value

                if t_val_u8 == dist && m_vec[i] != F::zero() {
                    found = true;
                    break;
                }
            }

            if !found {
                println!("  ⚠️  Value {dist} not found with non-zero count in lookup table.");
                _lookup_verification_passed = false;
            }
        }

        // Create the final MLEs for T and m.
        let t_mle = DenseMultilinearExtension::from_evaluations_vec(m_vars, t_vec);
        let m_mle = DenseMultilinearExtension::from_evaluations_vec(m_vars, m_vec);

        Ok((t_mle, m_mle))
    }

    /// Computes the Kronecker delta function `eq(x,z)` as an MLE.
    /// `eq(x,z) = Π_i (x_i * z_i + (1-x_i) * (1-z_i))`.
    fn compute_eq_mle(z: &[F]) -> Result<Rc<DenseMultilinearExtension<F>>, crate::Error> {
        let num_vars = z.len();
        let evals: Vec<F> = (0..(1 << num_vars))
            .into_par_iter()
            .map(|i| {
                let mut prod = F::one();
                for (j, z_j) in z.iter().enumerate().take(num_vars) {
                    let x_j = F::from(((i >> j) & 1) as u64); // j-th bit of i
                                                              // Compute the term for the j-th variable.
                    prod *= x_j * *z_j + (F::one() - x_j) * (F::one() - *z_j);
                }
                prod
            })
            .collect();
        Ok(Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, evals,
        )))
    }

    // --- Helper Functions for Data Handling ---

    /// Reads a file containing hexadecimal hash strings, one per line, and returns them as u64.
    fn read_hash_file(file_path: &str) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        reader
            .lines()
            .map(|line| {
                let hex_str = line?.trim().to_string();
                if hex_str.is_empty() {
                    return Ok(None); // Skip empty lines
                }
                // Parse hex string to u64.
                u64::from_str_radix(&hex_str, 16)
                    .map(Some) // Wrap result in Some
                    .map_err(Into::into) // Convert error type
            })
            .filter_map(Result::transpose) // Filter out None results and propagate errors
            .collect()
    }

    /// Converts a 64-bit hash into a 64-element bit vector (LSB first).
    fn hash_to_bit_vector(hash: u64) -> Vec<u8> {
        (0..64).map(|i| ((hash >> i) & 1) as u8).collect()
    }

    /// Computes Hamming distances between vector `a` and each row in `b_matrix`.
    fn compute_hamming_distances(a: &[u8], b_matrix: &[Vec<u8>]) -> Vec<u8> {
        b_matrix
            .par_iter()
            .map(|b_row| {
                // Hamming distance is the sum of bitwise XOR.
                a.iter()
                    .zip(b_row.iter())
                    .map(|(&a_bit, &b_bit)| a_bit ^ b_bit) // XOR bits
                    .sum() // Sum the differing bits
            })
            .collect()
    }

    /// Calculates ceiling of log base 2 of n. Equivalent to number of bits needed.
    fn log2_ceil(n: usize) -> usize {
        if n == 0 {
            0
        } else {
            // (n - 1).ilog2() gives floor(log2(n-1)). Add 1 for ceiling.
            (n - 1).ilog2() as usize + 1
        }
    }

    /// Flattens the B matrix into a single vector for MLE construction.
    /// The order interleaves the `y` (m variables) and `x` (n variables) dimensions.
    /// Indexing: `flat_idx = y_idx * (2^n) + x_idx`.
    fn flatten_b_matrix(b_matrix: &[Vec<u8>], n: usize, m: usize) -> Vec<u8> {
        let total_size = 1 << (n + m);
        let mut result = vec![0u8; total_size];
        for (i, row) in b_matrix.iter().enumerate() {
            for (j, &bit) in row.iter().enumerate() {
                // Calculate the flat index corresponding to y=i and x=j.
                let index = i * (1 << n) + j;
                if index < result.len() {
                    result[index] = bit;
                }
            }
        }
        result
    }

    /// Performs partial evaluation of an MLE `mle` at point `r`.
    /// This is equivalent to linear interpolation along the dimensions specified by `r`.
    /// The result is an MLE over the remaining variables.
    fn partial_evaluate(mle: &DenseMultilinearExtension<F>, r: &[F]) -> Vec<F> {
        let k = mle.num_vars(); // Total variables in the MLE.
        let m = r.len(); // Number of variables to evaluate at.
        assert!(
            k >= m,
            "Evaluation point dimension cannot exceed MLE dimension."
        );

        let mut evals = mle.evaluations.clone(); // Start with all evaluations.

        // Iterate through the evaluation dimensions specified by `r`.
        for i in 0..m {
            let r_val = r[m - 1 - i]; // Get the evaluation value for the current dimension (from last to first).
            let current_size = evals.len();
            let half_size = current_size / 2; // Size of the next-level MLE.

            // Perform linear interpolation in parallel.
            let next_evals: Vec<F> = (0..half_size)
                .into_par_iter()
                .map(|j| {
                    // Evaluate at index `j` using the formula:
                    // eval(j) = eval_even(j) * (1 - r_val) + eval_odd(j) * r_val
                    evals[j] * (F::one() - r_val) + evals[j + half_size] * r_val
                })
                .collect();
            evals = next_evals; // Update evaluations for the next dimension.
        }
        evals // Returns the evaluations of the resulting MLE.
    }

    /// Sets up the random weighting polynomial α(y) based on a single challenge `alpha`.
    /// It derives linear parameters `(a0_i, a1_i)` for each variable `i` and then computes α(y) evaluations.
    /// NOTE: This method is primarily for testing. In production, a proper Fiat-Shamir
    /// transcript derivation should be used for `alpha_linear_params`.
    fn set_alpha(&mut self, alpha: F) {
        let m = self.d_mle.num_vars();
        let num_evals = 1 << m;

        // Derive linear parameters (a0_i, a1_i) for α(y) = Π_i (a0_i + a1_i * y_i).
        // For testing, we use deterministic parameters based on alpha and index i.
        let mut a0 = Vec::with_capacity(m);
        let mut a1 = Vec::with_capacity(m);
        let alpha_sq = alpha * alpha;
        for i in 0..m {
            a0.push(alpha + F::from(i as u64)); // a0_i = alpha + i
            a1.push(alpha_sq + F::from(i as u64)); // a1_i = alpha^2 + i
        }
        self.alpha_linear_params = Some(a0.iter().cloned().zip(a1.iter().cloned()).collect());

        // Compute evaluations of α(y) = Π_i (a0_i + a1_i * y_i) over {0,1}^m.
        let alpha_values: Vec<F> = (0..num_evals)
            .into_par_iter()
            .map(|mask| {
                let mut prod = F::one();
                for i in 0..m {
                    let yi = (mask >> i) & 1; // Get the i-th bit of the current y evaluation point.
                                              // Calculate the term for the i-th variable: a0_i + a1_i * y_i.
                    let term = if yi == 0 { a0[i] } else { a0[i] + a1[i] };
                    prod *= term;
                }
                prod
            })
            .collect();
        // Store the computed evaluations as an MLE.
        self.alpha_mle = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            m,
            alpha_values,
        ));
    }

    /// Computes α(r) in O(m) time, where α(y)=∏_{i}(a0_i + a1_i·y_i).
    /// This uses the derived linear parameters.
    #[allow(dead_code)]
    fn alpha_value_at(&self, r: &[F]) -> Option<F> {
        let params = self.alpha_linear_params.as_ref()?; // Get the stored parameters.
        if params.len() != r.len() {
            return None; // Dimension mismatch.
        }
        let mut prod = F::one();
        // Compute the product Π_i (a0_i + a1_i * r_i).
        for (i, &(a0, a1)) in params.iter().enumerate() {
            prod *= a0 + a1 * r[i];
        }
        Some(prod)
    }

    /// Directly computes α(r) using the challenge `alpha` and the point `r`.
    /// Assumes the specific derivation rule: (a0_i, a1_i) = (α+i, α^2+i).
    fn alpha_value_at_from_challenge(alpha: F, r: &[F]) -> F {
        let alpha_sq = alpha * alpha;
        let mut prod = F::one();
        for (i, ri) in r.iter().enumerate() {
            let a0 = alpha + F::from(i as u64); // Derivation rule for a0_i
            let a1 = alpha_sq + F::from(i as u64); // Derivation rule for a1_i
            prod *= a0 + a1 * *ri; // Compute term and multiply into product
        }
        prod
    }

    /// Generates the complete `FullProof` by executing both phases and performing necessary commitments and openings.
    pub fn generate_full_proof(
        &mut self,
        alpha: F,  // Phase 1 challenge
        gamma: F,  // Phase 2 challenge
        lambda: F, // Phase 2 challenge
        r: &[F],   // Phase 1 random point (for y-domain)
        z: &[F],   // Phase 2 random point
    ) -> Result<FullProof<F, G>, crate::Error> {
        // --- Phase 1: Prove Consistency ---
        // Construct the witness polynomial g(y).
        self.construct_witness_g_polynomial(alpha)?;
        // Prove Σ g(y) = 0.
        let g_proof = self.prove_g_sum_zero()?;
        // Prove the evaluation f(r) and get the Sumcheck proof.
        let (f_r_val, f_proof) = self.prove_f_evaluation_at_r(r)?;

        // --- Phase 2: Prove Range ---
        // Prove Σ Q(y) = 0 for the merged range polynomial.
        let range_proof = self.prove_range_validity(gamma, lambda, z)?;

        // --- Hyrax Commitments ---
        let mut rng = ark_std::test_rng();
        let sponge = PoseidonSponge::<F>::new(&self.poseidon_config);

        // Commit to d̃(y)
        let (d_mle_padded, d_padded) = pad_mle_to_even_vars(&self.d_mle);
        let d_poly = LabeledPolynomial::new(
            "d".to_string(),
            (*d_mle_padded.as_ref()).clone(), // Padded polynomial
            None,                             // No transcript for commitment phase
            None,
        );
        let (d_coms, d_states) = HyraxPC::commit(
            self.ck_d.as_ref().unwrap(), // Committer key for m-vars
            &[d_poly.clone()],
            Some(&mut rng),
        )?;
        let d_comm = d_coms[0].commitment().clone();
        // Store commitment state and padding info for later openings.
        self.commitment_states
            .insert("d", (d_states[0].clone(), d_padded));

        // Commit to m̃(y_low) after lifting.
        if self.lifted_m_mle.is_none() {
            self.lift_lookup_to_m_domain();
        }
        let lifted_m = self.lifted_m_mle.as_ref().unwrap();
        let (m_mle_padded, m_padded) = pad_mle_to_even_vars(lifted_m);
        let m_poly = LabeledPolynomial::new(
            "m".to_string(),
            (*m_mle_padded.as_ref()).clone(),
            None,
            None,
        );
        let (m_coms, m_states) = HyraxPC::commit(
            self.ck_m.as_ref().unwrap(), // Committer key for m-vars
            &[m_poly.clone()],
            Some(&mut rng),
        )?;
        let m_comm = m_coms[0].commitment().clone();
        self.commitment_states
            .insert("m", (m_states[0].clone(), m_padded));

        // Commit to ã(x)
        let (a_mle_padded, a_padded) = pad_mle_to_even_vars(&self.a_mle);
        let a_poly = LabeledPolynomial::new(
            "a".to_string(),
            (*a_mle_padded.as_ref()).clone(),
            None,
            None,
        );
        let (a_coms, a_states) = HyraxPC::commit(
            self.ck_a.as_ref().unwrap(), // Committer key for n-vars
            &[a_poly.clone()],
            Some(&mut rng),
        )?;
        let a_comm = a_coms[0].commitment().clone();
        self.commitment_states
            .insert("a", (a_states[0].clone(), a_padded));

        // Commit to B̃(r_y, x) (partially evaluated B)
        let b_ry_mle = self.evaluate_b_at_r_y_as_mle_over_x(r);
        let (b_ry_mle_padded, b_padded) = pad_mle_to_even_vars(&b_ry_mle);
        let b_poly = LabeledPolynomial::new(
            "b".to_string(),
            (*b_ry_mle_padded.as_ref()).clone(),
            None,
            None,
        );
        let (b_coms, b_states) = HyraxPC::commit(
            self.ck_b.as_ref().unwrap(), // Committer key for n-vars
            &[b_poly.clone()],
            Some(&mut rng),
        )?;
        let b_comm = b_coms[0].commitment().clone();
        self.commitment_states
            .insert("b", (b_states[0].clone(), b_padded));

        // --- Hyrax Openings ---

        // Open d̃(y) at point r_y.
        let (d_at_r, proof_open_d_at_r) = {
            let (st, padded) = self.commitment_states.get("d").unwrap(); // Get commitment state and padding info
            let r_y_padded = pad_point(r, *padded); // Pad the evaluation point r if needed
            let d_lc = LabeledCommitment::new("d".to_string(), d_comm.clone(), None); // Labeled commitment
            let mut sp = sponge.clone(); // Use a cloned sponge for the opening process
            let batch = HyraxPC::open(
                self.ck_d.as_ref().unwrap(), // Committer key needed for opening
                &[d_poly.clone()],           // Polynomials to open
                &[d_lc],                     // Labeled commitments
                &r_y_padded,                 // Evaluation point
                &mut sp,                     // Sponge for transcript
                &[st.clone()],               // Commitment state
                Some(&mut rng),              // Random number generator
            )?;
            let r_vec: Vec<F> = r.to_vec(); // Convert r to Vec<F> for evaluation
            let d_at_r = self.d_mle.evaluate(&r_vec); // Evaluate d̃(r) locally
            (Some(d_at_r), Some(batch)) // Return evaluation and proof
        };

        // Determine the x-domain challenge point r_x from the f_proof's Sumcheck result.
        let n = self.a_mle.num_vars();
        // Reconstruct the polynomial info used for f's Sumcheck to verify its result.
        let mut f_info_poly = ListOfProductsOfPolynomials::new(n);
        let b_r_mle_for_f_info = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            n,
            Self::partial_evaluate(&self.b_mle, r), // B̃(r,x)
        ));
        f_info_poly.add_product(vec![self.a_mle.clone()], F::one());
        f_info_poly.add_product(vec![b_r_mle_for_f_info.clone()], F::one());
        f_info_poly.add_product(
            vec![self.a_mle.clone(), b_r_mle_for_f_info.clone()],
            -F::from(2u64),
        );
        // Verify f's Sumcheck result to get the challenge point r_x.
        let sub_rx = MLSumcheck::verify(&f_info_poly.info(), f_r_val, &f_proof)?;
        let r_x = sub_rx.point.clone(); // The challenge point r_x

        // Open ã(x) and B̃(r_y, x) at r_x. These can potentially be opened in a single batch.
        let (a_at_rx, b_at_rx, proof_open_ab_at_rx) = {
            let (a_st, a_pad) = self.commitment_states.get("a").unwrap();
            let (b_st, _b_pad) = self.commitment_states.get("b").unwrap();
            let r_x_padded = pad_point(&r_x, *a_pad); // Pad r_x if needed
            let a_lc = LabeledCommitment::new("a".to_string(), a_comm.clone(), None);
            let b_lc = LabeledCommitment::new("b".to_string(), b_comm.clone(), None);
            let mut sp = sponge.clone();
            // Attempt to open both ã and B̃(r_y, .) at r_x in one batch.
            let batch = HyraxPC::open(
                self.ck_a.as_ref().unwrap(),       // Committer key for n-vars polynomials
                &[a_poly.clone(), b_poly.clone()], // Polynomials to open
                &[a_lc, b_lc],                     // Labeled commitments
                &r_x_padded,                       // Evaluation point r_x
                &mut sp,
                &[a_st.clone(), b_st.clone()], // Commitment states
                Some(&mut rng),
            )?;
            // Evaluate locally for the proof struct.
            let a_val = self.a_mle.evaluate(&r_x);
            let b_val = b_ry_mle.evaluate(&r_x);
            (Some(a_val), Some(b_val), Some(batch))
        };

        // --- Openings at r_q (Phase 2 challenge point) ---
        let range_sub = MLSumcheck::verify(&self.range_info, F::zero(), &range_proof)?;
        let r_q = range_sub.point; // Challenge point r_q from Phase 2 Sumcheck
        let h_mle = self.h_mle.as_ref().unwrap(); // Get h MLE (constructed earlier)

        // Open d̃(y), h̃(y), and m̃(y_low) at r_q. These can be opened in batches.
        let (h_comm, proof_open_dh_at_rq, d_at_rq, h_at_rq, m_at_rq);
        {
            // Commit to h̃(y) first if it hasn't been committed yet (needed for openings).
            let h_comm_val; // Temporary variable to hold the committed h.
            let (h_committed, batch_proof, dval, hval) = {
                // Pad h MLE and create labeled polynomial for commitment.
                let (h_mle_padded, h_padded) = pad_mle_to_even_vars(h_mle);
                let h_poly = LabeledPolynomial::new(
                    "h".to_string(),
                    (*h_mle_padded.as_ref()).clone(),
                    None,
                    None,
                );
                // Commit to h̃(y).
                let (h_coms, h_states) = HyraxPC::commit(
                    self.ck_d.as_ref().unwrap(), // Committer key for m-vars
                    &[h_poly.clone()],
                    Some(&mut rng),
                )?;
                h_comm_val = h_coms[0].commitment().clone(); // Store the commitment
                self.commitment_states
                    .insert("h", (h_states[0].clone(), h_padded)); // Store state

                // Retrieve states and padded points for opening d, h, m at r_q.
                let (d_st, d_pad) = self.commitment_states.get("d").unwrap();
                let (h_st, _h_pad) = self.commitment_states.get("h").unwrap();
                let (m_st, _m_pad) = self.commitment_states.get("m").unwrap();
                let rq_pad = pad_point(&r_q, *d_pad); // Pad r_q for d/h opening

                // Create labeled commitments for d, h, m.
                let d_lc = LabeledCommitment::new("d".to_string(), d_comm.clone(), None);
                let h_lc = LabeledCommitment::new("h".to_string(), h_comm_val.clone(), None);
                let m_lc = LabeledCommitment::new("m".to_string(), m_comm.clone(), None);

                let mut sp = sponge.clone();
                // Open d̃, h̃, and m̃ at r_q in a single batch.
                let batch = HyraxPC::open(
                    self.ck_d.as_ref().unwrap(), // Committer key for m-vars
                    &[d_poly.clone(), h_poly.clone(), m_poly.clone()], // Polynomials
                    &[d_lc, h_lc, m_lc],         // Labeled commitments
                    &rq_pad,                     // Evaluation point r_q (padded)
                    &mut sp,
                    &[d_st.clone(), h_st.clone(), m_st.clone()], // Commitment states
                    Some(&mut rng),
                )?;
                // Evaluate locally for the proof struct.
                (
                    h_comm_val,                // The commitment to h
                    batch,                     // The opening proof
                    self.d_mle.evaluate(&r_q), // d(r_q)
                    h_mle.evaluate(&r_q),      // h(r_q)
                )
            };
            h_comm = h_committed; // Assign the committed value
            proof_open_dh_at_rq = Some(batch_proof); // Assign the batch proof
            d_at_rq = Some(dval);
            h_at_rq = Some(hval);
            // Evaluate m(r_q) locally. Note: uses the lifted m MLE.
            m_at_rq = Some(lifted_m.evaluate(&r_q));
        }
        // The proof for m might be part of the previous batch, so set to None here if handled that way.
        let proof_open_m_at_rq = None;

        // Construct and return the final FullProof object.
        Ok(FullProof {
            g_proof,
            f_proof,
            r_vec: r.to_vec(),
            f_r_val,
            alpha,
            range_proof,
            gamma,
            lambda,
            z_vec: z.to_vec(),
            d_comm: Some(d_comm),
            m_comm: Some(m_comm),
            a_comm: Some(a_comm),
            b_comm: Some(b_comm),
            h_comm: Some(h_comm),
            d_at_r,
            a_at_rx,
            b_at_rx,
            d_at_rq,
            m_at_rq,
            h_at_rq,
            proof_open_d_at_r,
            // Reuse the proof_open_ab_at_rx for both a and b openings, as they were likely batched.
            proof_open_a_at_rx: proof_open_ab_at_rx.clone(),
            proof_open_b_at_rx: proof_open_ab_at_rx,
            proof_open_dh_at_rq,
            proof_open_m_at_rq,
        })
    }
}

impl<F: PrimeField + Absorb, G: AffineRepr<ScalarField = F>> HammingDistanceProof<F, G> {
    /// Evaluates B at a fixed y=r as an MLE over x. This is done by iterating through
    /// the B matrix structure and applying the evaluation `r` across the `y` dimensions.
    /// Essentially, computes B̃(r, x) for all x.
    fn evaluate_b_at_r_y_as_mle_over_x(&self, r_y: &[F]) -> Rc<DenseMultilinearExtension<F>> {
        let n = self.a_mle.num_vars(); // Number of x variables
        let m = self.d_mle.num_vars(); // Number of y variables
        assert_eq!(r_y.len(), m, "Dimension mismatch for evaluation point r_y");

        let cols = 1usize << n; // Number of evaluations for x dimension (2^n)
        let rows = 1usize << m; // Number of evaluations for y dimension (2^m)
        let table = &self.b_mle.evaluations; // Flattened evaluations of B̃(y,x)
        let mut out = vec![F::zero(); cols]; // Resulting MLE evaluations over x

        // Iterate over each x dimension (each column in the conceptual B matrix).
        out.par_iter_mut().enumerate().for_each(|(x_idx, slot)| {
            // Extract the column corresponding to the current x_idx from the flattened table.
            // This effectively gives us B̃(y, x_idx) for all y.
            let mut col = vec![F::zero(); rows];
            for y in 0..rows {
                col[y] = table[y * cols + x_idx];
            }

            // Perform partial evaluation of this column (which is now a function of y) at point r_y.
            // This uses the same logic as `partial_evaluate` but applied iteratively.
            let mut next_col = vec![F::zero(); rows];
            let mut curr_len = rows;
            for &ry in r_y {
                // Iterate through the evaluation point r_y
                let half = curr_len >> 1; // Size of the next level after interpolation
                                          // Perform parallel chunked interpolation.
                next_col[..half].par_chunks_mut(1024).enumerate().for_each(
                    |(chunk_i, dst_chunk)| {
                        let start = chunk_i * 1024;
                        for (offset, dst) in dst_chunk.iter_mut().enumerate() {
                            let k = start + offset;
                            if k >= half {
                                break;
                            } // Boundary check
                              // Linear interpolation: E = E_even * (1-r) + E_odd * r
                            let a = col[2 * k];
                            let b = col[2 * k + 1];
                            *dst = a + (b - a) * ry;
                        }
                    },
                );
                // Update `col` for the next iteration.
                std::mem::swap(&mut col, &mut next_col);
                curr_len = half; // Reduce effective length
            }
            // After iterating through all r_y dimensions, col[0] holds B̃(r_y, x_idx).
            *slot = col[0];
        });
        // Construct and return the MLE over x variables.
        Rc::new(DenseMultilinearExtension::from_evaluations_vec(n, out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::{Fr, G1Affine};
    use ark_ff::UniformRand;
    use ark_std::rand::Rng;
    #[allow(unused_imports)]
    use std::fs::File; // Ensure File is in scope if used directly

    // Helper function to create a test file with random hashes.
    fn create_test_hash_file(num_hashes: usize, path: &str) {
        let mut file = std::fs::File::create(path).unwrap();
        let mut rng = ark_std::test_rng();
        for _ in 0..num_hashes {
            let hash: u64 = rng.gen();
            use std::io::Write; // Import Write trait
            writeln!(file, "{:016X}", hash).unwrap(); // Write hash in hex format
        }
    }

    #[test]
    fn test_rejects_distance_below_threshold_at_prove_stage() {
        // Use specific Field and Curve types for the test.
        type F = Fr;
        type G = G1Affine;

        const NUM_HASHES: usize = 128; // Number of hashes in the B matrix
        const THRESHOLD: u8 = 10; // Minimum allowed Hamming distance
        let a_hash = 0xAAAAAAAAAAAAAAAA; // Example hash for 'a'
        let hash_file_path = "/tmp/test_hashes_invalid_range_prove_fail.txt"; // Path for temporary hash file
        create_test_hash_file(NUM_HASHES, hash_file_path); // Create the test file

        // 1. Setup a malicious Prover by creating the HammingDistanceProof instance.
        let mut malicious_prover =
            HammingDistanceProof::<F, G>::new(a_hash, hash_file_path, THRESHOLD).unwrap();

        // 2. Simulate an attack: tamper with the witness data (`d_vector`).
        let invalid_distance = THRESHOLD - 5; // Inject a distance below the threshold.
        println!(
            "Injecting invalid distance: {}. Allowed range is [{}, 64].",
            invalid_distance, THRESHOLD
        );

        // Tamper the d_vector at index 0.
        malicious_prover.d_vector[0] = F::from(invalid_distance as u64);

        // Crucially, update the d_mle to reflect the tampered d_vector.
        // Otherwise, the MLE would still represent the original correct distances.
        let m_vars = malicious_prover.d_mle.num_vars();
        malicious_prover.d_mle = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            m_vars,
            malicious_prover.d_vector.clone(),
        ));

        // 3. Attempt to generate a malicious proof. This should fail because the witness is inconsistent
        //    with the claimed properties (e.g., the range constraint check during proof generation or verification).
        let mut rng = ark_std::test_rng();
        let alpha: F = F::rand(&mut rng); // Random challenge for Phase 1
        let gamma: F = F::rand(&mut rng); // Random challenge for Phase 2
        let lambda: F = F::rand(&mut rng); // Random challenge for Phase 2
        let r_y: Vec<F> = (0..m_vars).map(|_| F::rand(&mut rng)).collect(); // Random point for Phase 1
        let z: Vec<F> = (0..m_vars).map(|_| F::rand(&mut rng)).collect(); // Random point for Phase 2

        println!("Malicious prover is trying to generate a proof...");
        let proof_generation_result =
            malicious_prover.generate_full_proof(alpha, gamma, lambda, &r_y, &z);

        // 4. Assert that proof generation fails.
        assert!(
            proof_generation_result.is_err(),
            "Proof generation should fail when the witness is inconsistent!"
        );

        // Optionally, print the error message for debugging.
        if let Err(e) = proof_generation_result {
            println!("Proof generation failed as expected with error: {:?}", e);
            // More robust tests could check the specific error type or message.
            // For instance, checking if the error indicates inconsistency.
        }

        println!("✅ Test passed: Prover failed to generate an invalid proof, as expected.");
    }
}
