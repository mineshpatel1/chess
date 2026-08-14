//! The ladder driver's invariants: batching is only a speed change, every opening is played both
//! ways, and the tally it reports matches an independent, unbatched reference game for game.
//!
//! The reference in `unbatched` is deliberately not `LadderMatch` run at `in_flight: 1` - it is a
//! second, much simpler implementation built directly from `Search` and `best_move`, the same
//! differential-test shape `tests/alphabeta.rs` uses against plain negamax. A bug in the driver's
//! bookkeeping (queueing, retiring, which slot a batch answer belongs to) would show up as a
//! disagreement with this even though both ultimately call the same tree and the same alpha-beta.

use c4_core::connect4::Connect4;
use c4_core::encode::{PLANE_BYTES, PLANE_SIZE, POLICY_SIZE};
use c4_core::ladder::{Config, LadderMatch, Tally};
use c4_core::mcts::State;
use c4_core::search::best_move;
use c4_core::{PyRandom, Search};

include!("fixtures/ladder_fixture.rs");

/// SplitMix64, and the same stand-in shape `tests/selfplay.rs` uses: no network and no torch
/// needed to exercise the driver's mechanics.
fn mix(value: u64) -> u64 {
    let mut value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// A stand-in for the network: priors and a value that depend only on the position, the same
/// shape `evaluate_batch` hands back.
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

/// A single position's evaluation, for the unbatched reference below.
fn evaluate_one(board: &Connect4) -> ([f64; POLICY_SIZE], f64) {
    let mut planes = [0u8; PLANE_BYTES];
    c4_core::encode::planes(board, &mut planes);
    let mut legal = [false; POLICY_SIZE];
    c4_core::encode::legal_mask(board, &mut legal);

    let (priors, values) = evaluate(&planes, &legal);
    let mut widened = [0.0f64; POLICY_SIZE];
    for (slot, &prior) in priors.iter().enumerate() {
        widened[slot] = prior as f64;
    }
    (widened, values[0] as f64)
}

/// The evaluator `tests/zero/test_ladder_fast.py::write_fixture` pins its answers with - a hash
/// of the two bitboards directly, the same one `tests/search.rs` and `test_fast.py::hashed` use
/// for the MCTS fixture. Bitboard-based rather than plane-based like `evaluate` above, because the
/// Python side generating the fixture has the bitboards to hand and no reason to route through an
/// encoder to get a number both languages already agree on.
fn hashed_evaluate(board: &Connect4) -> ([f64; POLICY_SIZE], f64) {
    const SCALE: f64 = 9007199254740992.0; // 2^53, so the division is exact in both languages
    let base = mix(board.discs(board.turn()) ^ mix(board.discs(!board.turn())));

    let mut priors = [0.0; POLICY_SIZE];
    for (column, prior) in priors.iter_mut().enumerate() {
        *prior = (mix(base.wrapping_add(column as u64)) >> 11) as f64 / SCALE;
    }
    let value = (mix(base.wrapping_add(97)) >> 11) as f64 / SCALE * 2.0 - 1.0;
    (priors, value)
}

/// Deterministic, seedable "random" openings - a small LCG, the same shape `tests/alphabeta.rs`
/// uses, so a fuzz test is reproducible without pulling in a crate for it.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, bound: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % bound
    }
}

fn openings(count: usize, seed: u64) -> Vec<Connect4> {
    let mut rng = Lcg(seed);
    let mut found = Vec::with_capacity(count);
    while found.len() < count {
        let mut board = Connect4::new();
        let plies = 2 + rng.next(10);
        for _ in 0..plies {
            if board.is_game_over() {
                break;
            }
            let mut moves = [0u8; POLICY_SIZE];
            let legal = board.legal_moves(&mut moves);
            board.make_move(moves[rng.next(legal)]);
        }
        if !board.is_game_over() {
            found.push(board);
        }
    }
    found
}

fn config(depth: i32, simulations: u32, in_flight: usize) -> Config {
    Config { depth, simulations, exploration: 1.5, in_flight }
}

/// One game's moves and how it ended - `Record` without the borrow, for comparing.
type Played = (Vec<u8>, Option<bool>, bool);

/// Plays a whole rung, answering every batch it asks about.
fn play(openings: Vec<Connect4>, config: Config) -> Vec<Played> {
    let mut driver = LadderMatch::new(openings, config);
    while let Some(batch) = driver.pending() {
        let (priors, values) = evaluate(batch.planes, batch.legal);
        driver.submit(&priors, &values);
    }
    driver
        .records()
        .map(|record| (record.moves.clone(), record.winner, record.challenger_is_first))
        .collect()
}

/// One game, played straight through with no batching at all - `Search` and `best_move` called
/// directly, nothing borrowed from `ladder.rs`. Takes the evaluator as a parameter so the same
/// reference serves both the batching-invariance test (the plane-based `evaluate_one`) and the
/// pinned-fixture test (the bitboard-based `hashed_evaluate`).
fn play_one_unbatched(
    opening: &Connect4,
    challenger_is_first: bool,
    config: &Config,
    evaluate: impl Fn(&Connect4) -> ([f64; POLICY_SIZE], f64),
) -> Played {
    let mut board = opening.clone();
    let mut moves = Vec::new();
    let mut rng = PyRandom::from_text("unused: noise is off");

    if !board.is_game_over() && board.turn() != challenger_is_first {
        let column = best_move(&mut board, config.depth).expect("a legal move exists");
        board.make_move(column);
        moves.push(column);
    }

    while !board.is_game_over() {
        let mut tree = Search::new(config.simulations, config.exploration, 1.0, 0.25, false);
        tree.begin();
        while *tree.state() == State::NeedsEvaluation {
            let (priors, value) = evaluate(&board);
            tree.submit(&mut board, &priors, value, &mut rng);
        }
        let column = tree.result().most_visited();
        board.make_move(column);
        moves.push(column);

        if board.is_game_over() {
            break;
        }
        let opponent = best_move(&mut board, config.depth).expect("a legal move exists");
        board.make_move(opponent);
        moves.push(opponent);
    }

    (moves, board.result().expect("the game is over").winner, challenger_is_first)
}

#[test]
fn batching_does_not_change_the_games() {
    let config = config(3, 24, 1);
    let expected = play(openings(6, 1), config);
    for in_flight in [2, 3, 8, 64] {
        assert_eq!(
            play(openings(6, 1), self::config(3, 24, in_flight)),
            expected,
            "{in_flight} games in flight played different games"
        );
    }
}

#[test]
fn the_driver_matches_an_independent_unbatched_reference() {
    let config = config(2, 16, 4);
    for opening in openings(8, 2) {
        for challenger_is_first in [true, false] {
            let expected = play_one_unbatched(&opening, challenger_is_first, &config, evaluate_one);
            let got = play(vec![opening.clone()], self::config(2, 16, 4))
                .into_iter()
                .find(|(_, _, first)| *first == challenger_is_first)
                .expect("both colours were queued");
            assert_eq!(got, expected, "challenger_is_first={challenger_is_first}");
        }
    }
}

#[test]
fn challenger_matches_the_pinned_python_answers() {
    for &(columns, depth, exploration, simulations, moves, winner, challenger_is_first) in FIXTURE
    {
        let opening = Connect4::from_columns(columns);
        let config = Config { depth, simulations, exploration, in_flight: 1 };
        let got = play_one_unbatched(&opening, challenger_is_first, &config, hashed_evaluate);
        assert_eq!(
            got,
            (moves.to_vec(), winner, challenger_is_first),
            "{columns:?} depth {depth} at {simulations} simulations, challenger_is_first={challenger_is_first}"
        );
    }
}

#[test]
fn every_opening_is_played_with_both_colours() {
    let starts = openings(5, 3);
    let played = play(starts.clone(), config(2, 8, 6));
    assert_eq!(played.len(), starts.len() * 2);

    for pair in played.chunks(2) {
        let [first, second] = pair else { unreachable!() };
        assert!(first.2, "the first game of a pair should have the challenger first");
        assert!(!second.2, "the second game of a pair should have the challenger second");
    }
}

#[test]
fn the_tally_matches_the_records() {
    let starts = openings(10, 4);
    let mut driver = LadderMatch::new(starts, config(2, 12, 5));
    while let Some(batch) = driver.pending() {
        let (priors, values) = evaluate(batch.planes, batch.legal);
        driver.submit(&priors, &values);
    }

    let mut expected = Tally::default();
    for record in driver.records() {
        match record.winner {
            None => expected.draws += 1,
            Some(winner) if winner == record.challenger_is_first => expected.wins += 1,
            Some(_) => expected.losses += 1,
        }
    }
    assert_eq!(driver.tally(), expected);
    assert_eq!(driver.tally().wins + driver.tally().draws + driver.tally().losses, 20);
}

#[test]
fn an_opening_already_over_is_retired_without_any_search() {
    // Four in a column: whoever is about to move inherits a position that is already lost, and
    // there is nothing left to play, batched or not.
    let mut board = Connect4::new();
    for _ in 0..3 {
        board.make_move(0);
        board.make_move(1);
    }
    board.make_move(0);
    assert!(board.is_game_over());

    let mut driver = LadderMatch::new(vec![board], config(2, 8, 4));
    assert!(driver.pending().is_none(), "an already-over opening needs no evaluation");
    assert_eq!(driver.completed(), 2);
}
