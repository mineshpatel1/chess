//! The search against `ai/zero/mcts.py`, visit count for visit count.
//!
//! Both sides are handed the same deterministic evaluator - a hash of the position, so it needs
//! no network and no torch - and must produce the same tree. Nothing here is a tolerance: a PUCT
//! term computed in a different order, children in a different order, or a tie broken the other
//! way all show up as a different number.
//!
//! The expected counts were produced by the Python search and are pinned the way
//! `tests/connect4/solved.py` pins the solver's answers. `tests/zero/test_fast.py` runs the same
//! comparison live, so this file cannot quietly drift away from the implementation it copies.

use c4_core::constants::COLS;
use c4_core::encode::POLICY_SIZE;
use c4_core::mcts::State;
use c4_core::{Connect4, PyRandom, Search};

include!("fixtures/search_fixture.rs");

/// SplitMix64, and the same in `tests/zero/test_fast.py`. Anything both languages compute
/// exactly would do; this one spreads its output over the whole action space.
fn mix(value: u64) -> u64 {
    let mut value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// A stand-in for a network: priors and a value that depend only on the position.
fn evaluate(board: &Connect4) -> ([f64; POLICY_SIZE], f64) {
    const SCALE: f64 = 9007199254740992.0; // 2^53, so the division is exact in both languages
    let base = mix(board.discs(board.turn()) ^ mix(board.discs(!board.turn())));

    let mut priors = [0.0; POLICY_SIZE];
    for (column, prior) in priors.iter_mut().enumerate() {
        *prior = (mix(base.wrapping_add(column as u64)) >> 11) as f64 / SCALE;
    }
    let value = (mix(base.wrapping_add(97)) >> 11) as f64 / SCALE * 2.0 - 1.0;
    (priors, value)
}

/// Runs a whole search, answering every position it asks about.
fn search(columns: &[u8], exploration: f64, simulations: u32) -> ([u32; POLICY_SIZE], u8) {
    let mut board = Connect4::from_columns(columns);
    let mut tree = Search::new(simulations, exploration, 1.0, 0.25, false);
    let mut rng = PyRandom::from_text("noise is off, so this is never drawn from");

    tree.begin();
    while *tree.state() == State::NeedsEvaluation {
        let (priors, value) = evaluate(&board);
        tree.submit(&mut board, &priors, value, &mut rng);
    }

    let result = tree.result();
    let mut visits = [0u32; POLICY_SIZE];
    for slot in 0..result.children {
        visits[result.columns[slot] as usize] = result.visits[slot];
    }
    (visits, result.most_visited())
}

#[test]
fn visit_counts_match_the_python_search() {
    for &(columns, exploration, simulations, expected, best) in FIXTURE {
        let (visits, chosen) = search(columns, exploration, simulations);
        assert_eq!(
            visits, expected,
            "{columns:?} at c_puct {exploration}, {simulations} simulations"
        );
        assert_eq!(chosen, best, "{columns:?} chose a different move");
    }
}

#[test]
fn the_search_leaves_the_position_where_it_found_it() {
    for &(columns, exploration, simulations, _, _) in FIXTURE {
        let mut board = Connect4::from_columns(columns);
        let before = (board.discs(true), board.discs(false), board.turn(), board.ply());

        let mut tree = Search::new(simulations, exploration, 1.0, 0.25, false);
        let mut rng = PyRandom::from_text("unused");
        tree.begin();
        while *tree.state() == State::NeedsEvaluation {
            let (priors, value) = evaluate(&board);
            tree.submit(&mut board, &priors, value, &mut rng);
        }

        assert_eq!(
            (board.discs(true), board.discs(false), board.turn(), board.ply()),
            before,
            "{columns:?}"
        );
    }
}

#[test]
fn the_visits_add_up_to_the_simulations() {
    // The root is visited once by its own expansion and once per simulation; its children carry
    // every simulation that got past the root, which is all of them unless the game is over.
    for &(columns, exploration, simulations, _, _) in FIXTURE {
        let board = Connect4::from_columns(columns);
        if board.is_game_over() {
            continue;
        }

        let (visits, _) = search(columns, exploration, simulations);
        assert_eq!(
            visits.iter().sum::<u32>(),
            simulations,
            "{columns:?} at {simulations} simulations"
        );
    }
}

#[test]
fn a_full_column_is_never_searched() {
    let mut board = Connect4::new();
    for _ in 0..6 {
        board.make_move(3);
    }
    let mut moves = [0u8; COLS as usize];
    assert_eq!(board.legal_moves(&mut moves), 6);

    let (visits, _) = search(&[3, 3, 3, 3, 3, 3], 2.0, 600);
    assert_eq!(visits[3], 0);
}
