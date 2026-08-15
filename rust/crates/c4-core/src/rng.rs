//! CPython's `random.Random`, reimplemented so a Rust game can draw the same numbers.
//!
//! Self-play seeds a stream per game and draws from it for the root noise and for the move it
//! samples, so reproducing the stream is what lets the Rust engine play the *identical* games the
//! Python one plays rather than merely equivalent ones. Mersenne Twister, `random()`'s two
//! 32-bit draws, `gammavariate` and `choices` all follow CPython exactly; the comments say where
//! a line looks odd because CPython's does.

use sha2::{Digest, Sha512};

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

const LOG4: f64 = 1.386_294_361_119_890_6; // ln(4.0)
const SG_MAGICCONST: f64 = 2.504_077_396_776_274; // 1.0 + ln(4.5)

pub struct PyRandom {
    state: [u32; N],
    index: usize,
}

impl PyRandom {
    /// Seeded as `random.Random(text)` is: the utf-8 bytes and their SHA-512 digest read as one
    /// big-endian integer, which is then fed to `init_by_array` in 32-bit little-endian words.
    pub fn from_text(text: &str) -> Self {
        let mut generator = PyRandom {
            state: [0; N],
            index: N,
        };

        let mut digest = text.as_bytes().to_vec();
        digest.extend_from_slice(&Sha512::digest(text.as_bytes()));

        // The integer's bytes, least significant first, with the leading zeros of the big-endian
        // form dropped - `random_seed` keys off the number of bits, not the number of bytes.
        let mut little_endian: Vec<u8> = digest.into_iter().rev().collect();
        while little_endian.last() == Some(&0) {
            little_endian.pop();
        }

        let key: Vec<u32> = little_endian
            .chunks(4)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(0u32, |word, (byte, &value)| word | (value as u32) << (8 * byte))
            })
            .collect();

        generator.init_by_array(if key.is_empty() { &[0] } else { &key });
        generator
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for i in 1..N {
            let previous = self.state[i - 1];
            self.state[i] = 1812433253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);

        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            let previous = self.state[i - 1];
            self.state[i] = (self.state[i] ^ (previous ^ (previous >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.state[0] = self.state[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }

        for _ in 0..N - 1 {
            let previous = self.state[i - 1];
            self.state[i] = (self.state[i] ^ (previous ^ (previous >> 30)).wrapping_mul(1566083941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.state[0] = self.state[N - 1];
                i = 1;
            }
        }

        self.state[0] = 0x8000_0000;
        self.index = N;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }

        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    fn twist(&mut self) {
        for k in 0..N {
            let y = (self.state[k] & UPPER_MASK) | (self.state[(k + 1) % N] & LOWER_MASK);
            self.state[k] =
                self.state[(k + M) % N] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
        }
        self.index = 0;
    }

    /// A float in [0, 1), built from 53 bits as CPython's `random_random` builds it.
    pub fn random(&mut self) -> f64 {
        let high = (self.next_u32() >> 5) as f64;
        let low = (self.next_u32() >> 6) as f64;
        (high * 67108864.0 + low) * (1.0 / 9007199254740992.0)
    }

    fn getrandbits(&mut self, bits: u32) -> u32 {
        debug_assert!(bits <= 32);
        if bits == 0 {
            return 0;
        }
        self.next_u32() >> (32 - bits)
    }

    /// `random._randbelow`: reject until the draw lands under `bound`.
    pub fn below(&mut self, bound: u32) -> u32 {
        if bound == 0 {
            return 0;
        }
        let bits = u32::BITS - bound.leading_zeros();
        loop {
            let draw = self.getrandbits(bits);
            if draw < bound {
                return draw;
            }
        }
    }

    /// `random.gammavariate`. Self-play only ever asks for `alpha == 1.0`, which is the middle
    /// branch and a single logarithm; the other two are here so the parameter means what it says.
    pub fn gammavariate(&mut self, alpha: f64, beta: f64) -> f64 {
        assert!(alpha > 0.0 && beta > 0.0, "alpha and beta must be > 0.0");

        if alpha > 1.0 {
            let ainv = (2.0 * alpha - 1.0).sqrt();
            let bbb = alpha - LOG4;
            let ccc = alpha + ainv;
            loop {
                let u1 = self.random();
                if !(1e-7 < u1 && u1 < 0.9999999) {
                    continue;
                }
                let u2 = 1.0 - self.random();
                let v = (u1 / (1.0 - u1)).ln() / ainv;
                let x = alpha * v.exp();
                let z = u1 * u1 * u2;
                let r = bbb + ccc * v - x;
                if r + SG_MAGICCONST - 4.5 * z >= 0.0 || r >= z.ln() {
                    return x * beta;
                }
            }
        } else if alpha == 1.0 {
            -(1.0 - self.random()).ln() * beta
        } else {
            loop {
                let u = self.random();
                let b = (std::f64::consts::E + alpha) / std::f64::consts::E;
                let p = b * u;
                let x = if p <= 1.0 {
                    p.powf(1.0 / alpha)
                } else {
                    -((b - p) / alpha).ln()
                };
                let u1 = self.random();
                let accepted = if p > 1.0 {
                    u1 <= x.powf(alpha - 1.0)
                } else {
                    u1 <= (-x).exp()
                };
                if accepted {
                    return x * beta;
                }
            }
        }
    }

    /// `random.choices(population, weights, k=1)`: one draw against the running sums.
    pub fn weighted_index(&mut self, weights: &[f64]) -> usize {
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut running = 0.0;
        for &weight in weights {
            running += weight;
            cumulative.push(running);
        }

        let total = cumulative[cumulative.len() - 1];
        let target = self.random() * total;
        bisect_right(&cumulative, target, cumulative.len() - 1)
    }
}

/// `bisect.bisect_right(values, target, 0, hi)`.
fn bisect_right(values: &[f64], target: f64, hi: usize) -> usize {
    let (mut low, mut high) = (0usize, hi);
    while low < high {
        let middle = (low + high) / 2;
        if target < values[middle] {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    low
}
