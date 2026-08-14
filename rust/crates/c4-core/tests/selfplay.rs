//! The driver's invariants, chief among them that batching is only a speed change.

use c4_core::encode::{PLANE_BYTES, PLANE_SIZE, POLICY_SIZE};
use c4_core::selfplay::{Config, Record, SelfPlay};

fn mix(value: u64) -> u64 {
    let mut value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// A stand-in for the network: a masked softmax over logits that depend only on the position,
/// which is the shape `ai/zero/net.py::evaluate_batch` hands back.
fn evaluate(planes: &[u8], legal: &[bool]) -> (Vec<f32>, Vec<f32>) {
    let positions = legal.len() / POLICY_SIZE;
    let mut priors = vec![0.0f32; positions * POLICY_SIZE];
    let mut values = vec![0.0f32; positions];

    for position in 0..positions {
        let cells = &planes[position * PLANE_BYTES..(position + 1) * PLANE_BYTES];
        let mut key = 0u64;
        for (cell, &occupied) in cells.iter().enumerate() {
            if occupied != 0 {
                key ^= mix(cell as u64 + if cell >= PLANE_SIZE { 1 << 40 } else { 0 });
            }
        }

        let logits: Vec<f32> = (0..POLICY_SIZE)
            .map(|column| (mix(key.wrapping_add(column as u64)) >> 40) as f32 / 16777216.0 * 4.0)
            .collect();

        let masked: Vec<f32> = logits
            .iter()
            .zip(&legal[position * POLICY_SIZE..(position + 1) * POLICY_SIZE])
            .map(|(&logit, &legal)| if legal { logit } else { -1e9 })
            .collect();
        let highest = masked.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exponentials: Vec<f32> = masked.iter().map(|&l| (l - highest).exp()).collect();
        let total: f32 = exponentials.iter().sum();

        for column in 0..POLICY_SIZE {
            priors[position * POLICY_SIZE + column] = exponentials[column] / total;
        }
        values[position] = (mix(key.wrapping_add(97)) >> 40) as f32 / 16777216.0 * 2.0 - 1.0;
    }
    (priors, values)
}

fn config(games: usize, in_flight: usize, simulations: u32) -> Config {
    Config {
        games,
        in_flight,
        simulations,
        exploration: 2.0,
        dirichlet_alpha: 1.0,
        dirichlet_epsilon: 0.25,
        temperature: 1.0,
        final_temperature: 0.0,
        temperature_moves: 30,
        opening_plies: 0,
        seed: "1".to_string(),
    }
}

/// One game's examples: planes, policy targets, value targets, and how the game ended.
type Played = (Vec<u8>, Vec<f64>, Vec<f64>, Option<bool>);

/// Plays a whole generation, answering every batch it asks about.
fn play(config: Config) -> Vec<Played> {
    let mut driver = SelfPlay::new(config);
    while let Some(batch) = driver.pending() {
        let (priors, values) = evaluate(batch.planes, batch.legal);
        driver.submit(&priors, &values);
    }

    driver
        .records()
        .map(|record: &Record| {
            (
                record.planes.clone(),
                record.policies.clone(),
                record.values.clone(),
                record.winner,
            )
        })
        .collect()
}

#[test]
fn batching_does_not_change_the_games() {
    let expected = play(config(12, 1, 40));
    for in_flight in [2, 5, 12, 64] {
        assert_eq!(
            play(config(12, in_flight, 40)),
            expected,
            "{in_flight} games in flight played different games"
        );
    }
}

#[test]
fn a_game_records_one_example_per_ply_it_played() {
    let mut driver = SelfPlay::new(config(8, 8, 30));
    while let Some(batch) = driver.pending() {
        let (priors, values) = evaluate(batch.planes, batch.legal);
        driver.submit(&priors, &values);
    }

    assert_eq!(driver.completed(), 8);
    for record in driver.records() {
        assert_eq!(record.planes.len(), record.positions * PLANE_BYTES);
        assert_eq!(record.policies.len(), record.positions * POLICY_SIZE);
        assert_eq!(record.values.len(), record.positions);
        assert!((7..=42).contains(&record.positions), "{} plies", record.positions);
    }
    assert_eq!(driver.positions(), driver.records().map(|r| r.positions).sum());
}

#[test]
fn the_value_target_alternates_with_the_mover() {
    // A won game is +1 for the winner's positions and -1 for the loser's, taken in turn; a drawn
    // one is 0 throughout. Which player moved first in a position is its parity.
    for record in play(config(16, 16, 30)) {
        let (_, _, values, winner) = record;
        match winner {
            None => assert!(values.iter().all(|&value| value == 0.0)),
            Some(_) => {
                let last = *values.last().expect("a game has plies");
                for (ply, &value) in values.iter().rev().enumerate() {
                    let expected = if ply % 2 == 0 { last } else { -last };
                    assert_eq!(value, expected, "ply {ply} from the end");
                }
            }
        }
    }
}

#[test]
fn the_policy_target_is_a_distribution_over_legal_moves() {
    for (planes, policies, _, _) in play(config(8, 8, 40)) {
        for position in 0..policies.len() / POLICY_SIZE {
            let policy = &policies[position * POLICY_SIZE..(position + 1) * POLICY_SIZE];
            let cells = &planes[position * PLANE_BYTES..(position + 1) * PLANE_BYTES];
            let total: f64 = policy.iter().sum();
            assert!((total - 1.0).abs() < 1e-12, "policy sums to {total}");

            // A column is full when its top row holds a disc in either plane.
            for (column, &share) in policy.iter().enumerate() {
                let top = (POLICY_SIZE * 5) + column;
                let full = cells[top] != 0 || cells[PLANE_SIZE + top] != 0;
                assert!(!full || share == 0.0, "column {column} is full and scored {share}");
            }
        }
    }
}
