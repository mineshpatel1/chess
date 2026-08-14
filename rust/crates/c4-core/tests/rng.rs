//! The Mersenne Twister against CPython's, value for value.
//!
//! Every expected number below came out of `random.Random(seed)` in the interpreter. A stream
//! that agrees here is what lets the Rust engine play the same games the Python one plays, so
//! these are pinned to the bit rather than to a tolerance.

use c4_core::rng::PyRandom;

#[test]
fn random_matches_cpython() {
    let expected: [(&str, [f64; 6]); 3] = [
        (
            "1:0",
            [
                0.7259748788500867,
                0.419767232753784,
                0.37059490820682306,
                0.5262400788801375,
                0.8236857460417686,
                0.10477072290513978,
            ],
        ),
        (
            "1:37",
            [
                0.7442025600602974,
                0.4782769482576421,
                0.13049509354700906,
                0.8977487569560012,
                0.5708801043290478,
                0.5581923039043624,
            ],
        ),
        (
            "7:1999",
            [
                0.8885337048684433,
                0.11361280012183572,
                0.8301406120626023,
                0.4408061967664805,
                0.7937015998104273,
                0.171498899219931,
            ],
        ),
    ];

    for (seed, draws) in expected {
        let mut rng = PyRandom::from_text(seed);
        for (draw, &want) in draws.iter().enumerate() {
            assert_eq!(rng.random().to_bits(), want.to_bits(), "{seed} draw {draw}");
        }
    }
}

#[test]
fn gammavariate_matches_cpython_at_the_alpha_self_play_uses() {
    let expected: [(&str, [f64; 7]); 2] = [
        (
            "1:0",
            [
                1.2945354937678182,
                0.5443259331118397,
                0.4629802044684798,
                0.7470545811269648,
                1.7354873422610009,
                0.11067541798964516,
                0.7324390559517885,
            ],
        ),
        (
            "1:37",
            [
                1.3633693979418047,
                0.6506183841098322,
                0.13983130235956723,
                2.280322327223597,
                0.8460189220120999,
                0.8168805682844725,
                1.832743183998734,
            ],
        ),
    ];

    for (seed, draws) in expected {
        let mut rng = PyRandom::from_text(seed);
        for (draw, &want) in draws.iter().enumerate() {
            let got = rng.gammavariate(1.0, 1.0);
            assert_eq!(got.to_bits(), want.to_bits(), "{seed} draw {draw}");
        }
    }
}

#[test]
fn weighted_index_matches_choices() {
    // `random.choices([3,2,4,1,5,0,6], weights=w, k=1)`, reported as the index it landed on.
    let weights = [3.0, 17.0, 1.0, 40.0, 9.0, 2.0, 5.0];
    let expected = [3, 3, 3, 3, 4, 1, 3, 1, 3, 4];

    let mut rng = PyRandom::from_text("1:0");
    for (draw, &want) in expected.iter().enumerate() {
        assert_eq!(rng.weighted_index(&weights), want, "draw {draw}");
    }
}

#[test]
fn below_matches_randbelow() {
    // `random.choice` over seven moves, then `_randbelow(7)` from another stream.
    let mut rng = PyRandom::from_text("1:0");
    for (draw, &want) in [5, 6, 3, 2, 2, 4, 4, 6, 6, 3].iter().enumerate() {
        assert_eq!(rng.below(7), want, "choice draw {draw}");
    }

    let mut rng = PyRandom::from_text("5:5");
    for (draw, &want) in [1, 6, 4, 3, 0, 6, 1, 1, 2, 3].iter().enumerate() {
        assert_eq!(rng.below(7), want, "randbelow draw {draw}");
    }
}
