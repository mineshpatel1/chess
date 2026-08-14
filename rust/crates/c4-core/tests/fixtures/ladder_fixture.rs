// The Python reference's answers, generated once and pinned. Regenerate with
// `python3 -m tests.zero.test_ladder_fast --write-fixture`; see tests/ladder.rs for
// what they are compared against.

/// An opening, the rung played against it, and how the game came out.
pub type Pinned = (&'static [u8], i32, f64, u32, &'static [u8], Option<bool>, bool);

pub const FIXTURE: &[Pinned] = &[
    (&[4, 2, 1, 0, 3, 3], 2, 1.5, 24, &[0, 5, 0, 0, 2, 3, 1, 3, 4, 3], Some(false), true),
    (&[4, 2, 1, 0, 3, 3], 2, 1.5, 24, &[3, 1, 2, 2, 3, 0, 3, 4, 3], Some(true), false),
    (&[4, 2, 1, 0, 3, 3], 3, 2.0, 12, &[5, 6, 0, 4, 4, 2, 1, 5], Some(false), true),
    (&[4, 2, 1, 0, 3, 3], 3, 2.0, 12, &[3, 1, 2, 2, 3, 2, 3, 0, 3], Some(true), false),
    (&[6, 5, 3, 4, 5, 3], 2, 1.5, 24, &[5, 3, 4, 3, 1, 3], Some(false), true),
    (&[6, 5, 3, 4, 5, 3], 2, 1.5, 24, &[3, 3, 3, 5, 3, 6, 2, 6, 1, 4, 0], Some(true), false),
    (&[6, 5, 3, 4, 5, 3], 3, 2.0, 12, &[5, 3, 4, 5, 3, 4, 2, 4, 3, 2, 4, 1], Some(false), true),
    (&[6, 5, 3, 4, 5, 3], 3, 2.0, 12, &[4, 5, 4, 1, 2, 3, 3], Some(true), false),
    (&[5, 2, 3, 3, 1, 1], 2, 1.5, 24, &[1, 2, 5, 2, 2, 4, 3, 4], Some(false), true),
    (&[5, 2, 3, 3, 1, 1], 2, 1.5, 24, &[2, 1, 3, 6, 1, 0, 3, 2, 2, 5, 1, 1, 3, 2, 3], Some(true), false),
    (&[5, 2, 3, 3, 1, 1], 3, 2.0, 12, &[2, 3, 5, 3, 1, 3], Some(false), true),
    (&[5, 2, 3, 3, 1, 1], 3, 2.0, 12, &[2, 5, 3, 2, 4, 5, 6], Some(true), false),
    (&[3, 2, 3, 5, 1, 3], 2, 1.5, 24, &[2, 1, 4, 4, 5, 3, 5, 2, 6, 2], Some(false), true),
    (&[3, 2, 3, 5, 1, 3], 2, 1.5, 24, &[3, 5, 5, 5, 3, 1, 3, 1, 1, 4, 4, 6, 4, 4, 5, 6, 4, 5, 6, 0, 6], Some(true), false),
    (&[3, 2, 3, 5, 1, 3], 3, 2.0, 12, &[2, 2, 5, 1, 2, 3, 3, 0], Some(false), true),
    (&[3, 2, 3, 5, 1, 3], 3, 2.0, 12, &[2, 6, 1, 3, 1, 0, 1], Some(true), false),
    (&[6, 5, 3, 2, 0, 0], 2, 1.5, 24, &[6, 6, 0, 3, 1, 3, 0, 3, 3, 0, 5, 3, 6, 2, 0, 1], Some(false), true),
    (&[6, 5, 3, 2, 0, 0], 2, 1.5, 24, &[3, 3, 3, 3, 3, 2, 2, 5, 2, 2, 4, 6, 2, 4, 1, 1, 5, 5, 5, 5, 4, 1, 1], Some(true), false),
    (&[6, 5, 3, 2, 0, 0], 3, 2.0, 12, &[6, 3, 6, 6, 4, 3, 1, 2, 0, 1], Some(false), true),
    (&[6, 5, 3, 2, 0, 0], 3, 2.0, 12, &[3, 5, 3, 6, 3], Some(true), false),
    (&[5, 3, 5, 5, 1, 3], 2, 1.5, 24, &[2, 3, 2, 3], Some(false), true),
    (&[5, 3, 5, 5, 1, 3], 2, 1.5, 24, &[3, 6, 3, 1, 3, 0, 3], Some(true), false),
    (&[5, 3, 5, 5, 1, 3], 3, 2.0, 12, &[2, 2, 6, 1, 4, 4], Some(false), true),
    (&[5, 3, 5, 5, 1, 3], 3, 2.0, 12, &[3, 6, 3, 1, 3, 2, 3], Some(true), false),
];
