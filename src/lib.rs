//! `lau-simd-vibe` — low-level vibe field computation engine using SIMD-style packed operations.
//!
//! No unsafe code, no external SIMD crates. All packed operations are emulated with explicit
//! loops over fixed-size arrays.

use std::f64::consts::TAU;

// ---------------------------------------------------------------------------
// VibeField
// ---------------------------------------------------------------------------

/// A fixed-size field of `f64` values with an associated tick counter.
#[derive(Debug, Clone, PartialEq)]
pub struct VibeField<const N: usize> {
    pub values: [f64; N],
    pub tick: u64,
}

impl<const N: usize> VibeField<N> {
    /// Create a zero-initialized field at tick 0.
    pub fn zero() -> Self {
        Self {
            values: [0.0; N],
            tick: 0,
        }
    }

    /// Create a field from a closure `f(i) → value`.
    pub fn from_fn(f: impl Fn(usize) -> f64) -> Self {
        let mut values = [0.0; N];
        let mut i = 0;
        while i < N {
            values[i] = f(i);
            i += 1;
        }
        Self { values, tick: 0 }
    }

    /// Element-wise addition (packed-style).
    pub fn add(&self, other: &Self) -> Self {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            out[i] = self.values[i] + other.values[i];
            i += 1;
        }
        Self {
            values: out,
            tick: self.tick,
        }
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            out[i] = self.values[i] - other.values[i];
            i += 1;
        }
        Self {
            values: out,
            tick: self.tick,
        }
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Self) -> Self {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            out[i] = self.values[i] * other.values[i];
            i += 1;
        }
        Self {
            values: out,
            tick: self.tick,
        }
    }

    /// Scale every element by `s`.
    pub fn scale(&self, s: f64) -> Self {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            out[i] = self.values[i] * s;
            i += 1;
        }
        Self {
            values: out,
            tick: self.tick,
        }
    }

    /// Horizontal sum of all elements.
    pub fn sum(&self) -> f64 {
        let mut acc = 0.0;
        let mut i = 0;
        while i < N {
            acc += self.values[i];
            i += 1;
        }
        acc
    }

    /// Dot product with another field.
    pub fn dot(&self, other: &Self) -> f64 {
        let mut acc = 0.0;
        let mut i = 0;
        while i < N {
            acc += self.values[i] * other.values[i];
            i += 1;
        }
        acc
    }

    /// L2 norm (Euclidean length).
    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Return a unit-length copy. Returns `Self::zero()` if the norm is zero.
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n == 0.0 {
            return Self::zero();
        }
        self.scale(1.0 / n)
    }

    /// Maximum element value.
    pub fn max_value(&self) -> f64 {
        let mut m = self.values[0];
        let mut i = 1;
        while i < N {
            if self.values[i] > m {
                m = self.values[i];
            }
            i += 1;
        }
        m
    }

    /// Minimum element value.
    pub fn min_value(&self) -> f64 {
        let mut m = self.values[0];
        let mut i = 1;
        while i < N {
            if self.values[i] < m {
                m = self.values[i];
            }
            i += 1;
        }
        m
    }

    /// Relative conservation error compared to a baseline:
    /// `|sum(self) - sum(baseline)| / |sum(baseline)|`.
    ///
    /// Returns 0.0 when both sums are zero. Returns `f64::INFINITY` when baseline sums to
    /// zero but self does not.
    pub fn conservation_error(&self, baseline: &Self) -> f64 {
        let s = self.sum();
        let b = baseline.sum();
        if b == 0.0 && s == 0.0 {
            0.0
        } else if b == 0.0 {
            f64::INFINITY
        } else {
            (s - b).abs() / b.abs()
        }
    }

    /// Clamp every element to `[lo, hi]`.
    pub fn clamp(&self, lo: f64, hi: f64) -> Self {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            out[i] = if self.values[i] < lo {
                lo
            } else if self.values[i] > hi {
                hi
            } else {
                self.values[i]
            };
            i += 1;
        }
        Self {
            values: out,
            tick: self.tick,
        }
    }

    /// Linear interpolation: `self * (1 - t) + other * t`.
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let mut out = [0.0; N];
        let one_minus_t = 1.0 - t;
        let mut i = 0;
        while i < N {
            out[i] = one_minus_t * self.values[i] + t * other.values[i];
            i += 1;
        }
        Self {
            values: out,
            tick: self.tick,
        }
    }
}

// ---------------------------------------------------------------------------
// VibeKernel
// ---------------------------------------------------------------------------

/// A 1-D convolution kernel of fixed size `N`.
#[derive(Debug, Clone, PartialEq)]
pub struct VibeKernel<const N: usize> {
    pub weights: [f64; N],
}

impl<const N: usize> VibeKernel<N> {
    /// Gaussian kernel centered at `N / 2`, normalised to sum to 1.
    ///
    /// `sigma` is in units of array indices.
    pub fn gaussian(sigma: f64) -> Self {
        let center = (N - 1) as f64 / 2.0;
        let mut weights = [0.0; N];
        let mut i = 0;
        while i < N {
            let x = i as f64 - center;
            weights[i] = (-0.5 * (x / sigma).powi(2)).exp();
            i += 1;
        }
        let total: f64 = weights.iter().copied().sum();
        if total > 0.0 {
            let mut i = 0;
            while i < N {
                weights[i] /= total;
                i += 1;
            }
        }
        Self { weights }
    }

    /// Uniform box filter — every weight is `1 / N`.
    pub fn box_filter() -> Self {
        let w = 1.0 / N as f64;
        Self {
            weights: [w; N],
        }
    }

    /// Apply the kernel to a field (dot product = 1-D convolution at centre).
    pub fn apply(&self, field: &VibeField<N>) -> f64 {
        let mut acc = 0.0;
        let mut i = 0;
        while i < N {
            acc += self.weights[i] * field.values[i];
            i += 1;
        }
        acc
    }
}

// ---------------------------------------------------------------------------
// VibeDiffusion
// ---------------------------------------------------------------------------

/// Discrete 1-D diffusion operator with Neumann (clamped) boundaries.
pub struct VibeDiffusion;

impl VibeDiffusion {
    /// Single diffusion step:
    /// `new[i] = field[i] + rate * (field[i-1] - 2*field[i] + field[i+1])`
    ///
    /// Boundary conditions: indices outside `[0, N)` clamp to the nearest valid index.
    pub fn step<const N: usize>(&self, field: &VibeField<N>, rate: f64) -> VibeField<N> {
        let mut out = [0.0; N];
        let mut i = 0;
        while i < N {
            let prev = if i == 0 { field.values[0] } else { field.values[i - 1] };
            let next = if i == N - 1 {
                field.values[N - 1]
            } else {
                field.values[i + 1]
            };
            out[i] = field.values[i] + rate * (prev - 2.0 * field.values[i] + next);
            i += 1;
        }
        VibeField {
            values: out,
            tick: field.tick + 1,
        }
    }

    /// Run diffusion for `iterations` steps, returning the final field.
    pub fn steady_state<const N: usize>(
        &self,
        field: &VibeField<N>,
        rate: f64,
        iterations: usize,
    ) -> VibeField<N> {
        let mut current = field.clone();
        let mut step = 0;
        while step < iterations {
            current = self.step(&current, rate);
            step += 1;
        }
        current
    }
}

// ---------------------------------------------------------------------------
// VibeSpectrum
// ---------------------------------------------------------------------------

/// Frequency-domain analysis of vibe fields via naive DFT.
pub struct VibeSpectrum;

impl VibeSpectrum {
    /// Naive DFT magnitude spectrum — O(N²).
    ///
    /// Returns an array where entry `k` is `|X[k]| = sqrt(Re² + Im²)`.
    pub fn dft_magnitudes<const N: usize>(&self, field: &VibeField<N>) -> [f64; N] {
        let mut mags = [0.0; N];
        let mut k = 0;
        while k < N {
            let mut re = 0.0;
            let mut im = 0.0;
            let mut n = 0;
            while n < N {
                let angle = -TAU * (k as f64) * (n as f64) / (N as f64);
                re += field.values[n] * angle.cos();
                im += field.values[n] * angle.sin();
                n += 1;
            }
            mags[k] = re.hypot(im);
            k += 1;
        }
        mags
    }

    /// Return the index of the dominant frequency (largest magnitude, excluding DC).
    pub fn dominant_frequency<const N: usize>(&self, field: &VibeField<N>) -> usize {
        let mags = self.dft_magnitudes(field);
        // Skip k=0 (DC). If N < 2, return 0.
        if N < 2 {
            return 0;
        }
        let mut best_k = 1;
        let mut best_mag = mags[1];
        let mut k = 2;
        while k < N {
            if mags[k] > best_mag {
                best_mag = mags[k];
                best_k = k;
            }
            k += 1;
        }
        best_k
    }

    /// Sum of squared magnitudes (total spectral energy).
    pub fn spectral_energy<const N: usize>(&self, field: &VibeField<N>) -> f64 {
        let mags = self.dft_magnitudes(field);
        let mut energy = 0.0;
        let mut k = 0;
        while k < N {
            energy += mags[k] * mags[k];
            k += 1;
        }
        energy
    }
}

// ---------------------------------------------------------------------------
// VibeQuantize
// ---------------------------------------------------------------------------

/// Quantisation helpers for 8-bit packing of vibe fields.
pub struct VibeQuantize;

impl VibeQuantize {
    /// Map a single `f64` in `[min, max]` → `[0, 255]`.
    ///
    /// Values outside `[min, max]` are clamped.
    pub fn quantize_8bit(value: f64, min: f64, max: f64) -> u8 {
        let range = max - min;
        if range <= 0.0 {
            return 0;
        }
        let normalised = (value - min) / range;
        let clamped = normalised.clamp(0.0, 1.0);
        (clamped * 255.0).round() as u8
    }

    /// Map a `u8` back to `f64` in `[min, max]`.
    pub fn dequantize_8bit(byte: u8, min: f64, max: f64) -> f64 {
        min + (byte as f64 / 255.0) * (max - min)
    }

    /// Pack an entire field into `[u8; N]`.
    pub fn pack_field_8bit<const N: usize>(field: &VibeField<N>, min: f64, max: f64) -> [u8; N] {
        let mut packed = [0u8; N];
        let mut i = 0;
        while i < N {
            packed[i] = Self::quantize_8bit(field.values[i], min, max);
            i += 1;
        }
        packed
    }

    /// Unpack `[u8; N]` back into a `VibeField<N>`.
    pub fn unpack_field_8bit<const N: usize>(packed: &[u8; N], min: f64, max: f64) -> VibeField<N> {
        let mut values = [0.0; N];
        let mut i = 0;
        while i < N {
            values[i] = Self::dequantize_8bit(packed[i], min, max);
            i += 1;
        }
        VibeField { values, tick: 0 }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- VibeField basics ---

    #[test]
    fn zero_field_is_all_zeros() {
        let f: VibeField<8> = VibeField::zero();
        assert_eq!(f.values, [0.0; 8]);
        assert_eq!(f.tick, 0);
    }

    #[test]
    fn from_fn_identity() {
        let f: VibeField<4> = VibeField::from_fn(|i| i as f64);
        assert_eq!(f.values, [0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn add_element_wise() {
        let a: VibeField<4> = VibeField::from_fn(|i| i as f64);
        let b: VibeField<4> = VibeField::from_fn(|i| (i + 1) as f64);
        let c = a.add(&b);
        assert_eq!(c.values, [1.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn sub_element_wise() {
        let a: VibeField<4> = VibeField::from_fn(|i| (i + 2) as f64);
        let b: VibeField<4> = VibeField::from_fn(|i| i as f64);
        let c = a.sub(&b);
        assert_eq!(c.values, [2.0; 4]);
    }

    #[test]
    fn mul_element_wise() {
        let a: VibeField<3> = VibeField::from_fn(|i| (i + 1) as f64);
        let b: VibeField<3> = VibeField::from_fn(|i| (i + 1) as f64);
        let c = a.mul(&b);
        assert_eq!(c.values, [1.0, 4.0, 9.0]);
    }

    #[test]
    fn scale_doubles() {
        let a: VibeField<4> = VibeField::from_fn(|i| i as f64);
        let c = a.scale(2.0);
        assert_eq!(c.values, [0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn sum_works() {
        let f: VibeField<5> = VibeField::from_fn(|i| (i + 1) as f64);
        assert!((f.sum() - 15.0).abs() < 1e-12);
    }

    #[test]
    fn dot_product() {
        let a: VibeField<3> = VibeField::from_fn(|i| (i + 1) as f64);
        let b: VibeField<3> = VibeField::from_fn(|i| (i + 1) as f64);
        assert!((a.dot(&b) - 14.0).abs() < 1e-12); // 1+4+9
    }

    #[test]
    fn norm_unit_vector() {
        let f: VibeField<3> = VibeField::from_fn(|_| 1.0);
        let n = f.normalize();
        let expected = 1.0 / 3.0_f64.sqrt();
        for v in &n.values {
            assert!((v - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn max_min_values() {
        let f: VibeField<5> = VibeField::from_fn(|i| (i as f64 - 2.0)); // [-2,-1,0,1,2]
        assert!((f.max_value() - 2.0).abs() < 1e-12);
        assert!((f.min_value() - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn conservation_error_identical() {
        let f: VibeField<4> = VibeField::from_fn(|i| (i + 1) as f64);
        assert!((f.conservation_error(&f)).abs() < 1e-15);
    }

    #[test]
    fn conservation_error_shifted() {
        let baseline: VibeField<3> = VibeField::from_fn(|i| (i + 1) as f64); // sum = 6
        let shifted: VibeField<3> = VibeField::from_fn(|i| (i + 2) as f64); // sum = 9
        let err = shifted.conservation_error(&baseline);
        assert!((err - 0.5).abs() < 1e-12); // |9-6|/6 = 0.5
    }

    #[test]
    fn clamp_clamps() {
        let f: VibeField<4> = VibeField::from_fn(|i| (i as f64 - 1.5)); // [-1.5, -0.5, 0.5, 1.5]
        let c = f.clamp(-1.0, 1.0);
        assert_eq!(c.values[0], -1.0);
        assert_eq!(c.values[3], 1.0);
        assert!((c.values[1] - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn lerp_midpoint() {
        let a: VibeField<3> = VibeField::from_fn(|_| 0.0);
        let b: VibeField<3> = VibeField::from_fn(|_| 10.0);
        let mid = a.lerp(&b, 0.5);
        for v in &mid.values {
            assert!((v - 5.0).abs() < 1e-12);
        }
    }

    // --- VibeKernel ---

    #[test]
    fn gaussian_sums_to_one() {
        let k: VibeKernel<9> = VibeKernel::gaussian(1.5);
        let total: f64 = k.weights.iter().copied().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn box_filter_uniform() {
        let k: VibeKernel<4> = VibeKernel::box_filter();
        for w in &k.weights {
            assert!((w - 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn kernel_apply_dot_product() {
        let k: VibeKernel<3> = VibeKernel { weights: [1.0, 2.0, 3.0] };
        let f: VibeField<3> = VibeField::from_fn(|i| (i + 1) as f64);
        // 1*1 + 2*2 + 3*3 = 14
        assert!((k.apply(&f) - 14.0).abs() < 1e-12);
    }

    // --- VibeDiffusion ---

    #[test]
    fn diffusion_flat_field_unchanged() {
        let flat: VibeField<8> = VibeField::from_fn(|_| 5.0);
        let diff = VibeDiffusion;
        let result = diff.step(&flat, 0.25);
        // Second derivative of constant is 0
        for v in &result.values {
            assert!((v - 5.0).abs() < 1e-12);
        }
    }

    #[test]
    fn diffusion_conserves_energy() {
        let field: VibeField<16> = VibeField::from_fn(|i| {
            (2.0 * PI * i as f64 / 16.0).sin()
        });
        let diff = VibeDiffusion;
        let before = field.sum();
        let after = diff.step(&field, 0.1).sum();
        assert!((before - after).abs() < 1e-10, "before={before}, after={after}");
    }

    #[test]
    fn diffusion_steady_state_converges_to_mean() {
        let field: VibeField<8> = VibeField::from_fn(|i| (i as f64 * 10.0));
        let diff = VibeDiffusion;
        let result = diff.steady_state(&field, 0.2, 10_000);
        let mean = field.sum() / 8.0;
        for v in &result.values {
            assert!((v - mean).abs() < 1e-6, "v={v}, mean={mean}");
        }
    }

    #[test]
    fn diffusion_increments_tick() {
        let f: VibeField<4> = VibeField::from_fn(|_| 1.0);
        let diff = VibeDiffusion;
        assert_eq!(diff.step(&f, 0.1).tick, 1);
        assert_eq!(diff.steady_state(&f, 0.1, 5).tick, 5);
    }

    // --- VibeSpectrum ---

    #[test]
    fn dft_dc_signal() {
        let f: VibeField<8> = VibeField::from_fn(|_| 1.0);
        let spec = VibeSpectrum;
        let mags = spec.dft_magnitudes(&f);
        // DC component should be N, all others 0
        assert!((mags[0] - 8.0).abs() < 1e-10);
        for k in 1..8 {
            assert!(mags[k] < 1e-10, "mags[{k}] = {}", mags[k]);
        }
    }

    #[test]
    fn dft_single_frequency() {
        // Pure sine at bin 1
        let f: VibeField<16> = VibeField::from_fn(|i| (TAU * i as f64 / 16.0).sin());
        let spec = VibeSpectrum;
        let dom = spec.dominant_frequency(&f);
        assert_eq!(dom, 1);
    }

    #[test]
    fn spectral_energy_positive() {
        let f: VibeField<8> = VibeField::from_fn(|i| (i as f64).sin());
        let spec = VibeSpectrum;
        let e = spec.spectral_energy(&f);
        assert!(e > 0.0);
    }

    #[test]
    fn spectral_energy_parseval() {
        // By Parseval's theorem: sum |x[n]|² = (1/N) * sum |X[k]|²
        let f: VibeField<8> = VibeField::from_fn(|i| (i as f64 + 1.0));
        let spec = VibeSpectrum;
        let time_energy: f64 = f.values.iter().map(|v| v * v).sum();
        let spec_energy = spec.spectral_energy(&f);
        // spectral_energy returns sum |X[k]|², so spec_energy / N should ≈ time_energy
        let ratio = spec_energy / 8.0;
        assert!((ratio - time_energy).abs() < 1e-10, "ratio={ratio}, time={time_energy}");
    }

    // --- VibeQuantize ---

    #[test]
    fn quantize_roundtrip() {
        let value = 0.5;
        let byte = VibeQuantize::quantize_8bit(value, 0.0, 1.0);
        let recovered = VibeQuantize::dequantize_8bit(byte, 0.0, 1.0);
        assert!((recovered - value).abs() < 1.0 / 255.0);
    }

    #[test]
    fn quantize_clamps_out_of_range() {
        assert_eq!(VibeQuantize::quantize_8bit(-1.0, 0.0, 1.0), 0);
        assert_eq!(VibeQuantize::quantize_8bit(2.0, 0.0, 1.0), 255);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let field: VibeField<4> = VibeField::from_fn(|i| (i as f64 + 1.0) * 10.0);
        let packed = VibeQuantize::pack_field_8bit(&field, 10.0, 40.0);
        let unpacked = VibeQuantize::unpack_field_8bit(&packed, 10.0, 40.0);
        for i in 0..4 {
            let err = (unpacked.values[i] - field.values[i]).abs();
            assert!(err < 40.0 / 255.0, "idx={i}, err={err}");
        }
    }

    #[test]
    fn quantize_zero_range() {
        assert_eq!(VibeQuantize::quantize_8bit(42.0, 5.0, 5.0), 0);
    }

    // --- Edge cases ---

    #[test]
    fn normalize_zero_field_stays_zero() {
        let f: VibeField<3> = VibeField::zero();
        let n = f.normalize();
        assert_eq!(n.values, [0.0; 3]);
    }

    #[test]
    fn lerp_at_zero_and_one() {
        let a: VibeField<2> = VibeField::from_fn(|_| 0.0);
        let b: VibeField<2> = VibeField::from_fn(|_| 10.0);
        let at_zero = a.lerp(&b, 0.0);
        let at_one = a.lerp(&b, 1.0);
        assert!((at_zero.values[0]).abs() < 1e-12);
        assert!((at_one.values[0] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn conservation_error_both_zero() {
        let a: VibeField<4> = VibeField::zero();
        assert!((a.conservation_error(&a)).abs() < 1e-15);
    }

    #[test]
    fn diffusion_boundary_clamping() {
        // Spike at index 0 should not leak out
        let mut f: VibeField<4> = VibeField::zero();
        f.values[0] = 1.0;
        let diff = VibeDiffusion;
        let after = diff.step(&f, 0.25);
        let total_before = 1.0;
        let total_after: f64 = after.values.iter().copied().sum();
        assert!((total_before - total_after).abs() < 1e-12, "before={total_before}, after={total_after}");
    }

    #[test]
    fn gaussian_symmetric() {
        let k: VibeKernel<7> = VibeKernel::gaussian(1.0);
        // Should be symmetric: w[0] == w[6], w[1] == w[5], etc.
        assert!((k.weights[0] - k.weights[6]).abs() < 1e-15);
        assert!((k.weights[1] - k.weights[5]).abs() < 1e-15);
        assert!((k.weights[2] - k.weights[4]).abs() < 1e-15);
    }

    #[test]
    fn dominant_frequency_dc_only() {
        let f: VibeField<8> = VibeField::from_fn(|_| 3.0);
        let spec = VibeSpectrum;
        // With only DC, all non-DC bins are ~0 — any k in 1..N is valid
        let dom = spec.dominant_frequency(&f);
        assert!((1..8).contains(&dom), "dom={dom}");
    }

    #[test]
    fn diffusion_steady_state_tick() {
        let f: VibeField<4> = VibeField::from_fn(|i| i as f64);
        let diff = VibeDiffusion;
        let result = diff.steady_state(&f, 0.1, 42);
        assert_eq!(result.tick, 42);
    }
}
