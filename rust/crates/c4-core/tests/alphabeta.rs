//! The alpha-beta against an unpruned reference, and the evaluation against its own invariants.
//!
//! The argument is `tests/connect4/test_search_equivalence.py`'s and is repeated here because it
//! is worth having on this side of the boundary too: alpha-beta returns the *identical* result to
//! plain negamax, so a version that picks a different move is not a faster search, it is a broken
//! one - classically a sign error in the `-beta, -alpha` swap, which produces a search that is
//! still plausible, still fast and quietly worse. The reference is twenty lines and lives in the
//! test rather than in the crate, because a second search shipped beside the first is a second
//! search to maintain.
//!
//! That the answers also match *Python's* is `tests/connect4/test_native.py` and the pinned
//! fixture below.

use c4_core::connect4::DRAW;
use c4_core::constants::{COLS, COLUMN_MASKS, FULL_BOARD, RED, ROWS, STRIDE, YELLOW};
use c4_core::evaluation::{value, weighted_eval, MAX_EVAL};
use c4_core::search::{best_move, negamax_ab, root_scores, terminal_score, HIGH_BOUND, LOW_BOUND, MATE};
use c4_core::Connect4;

include!("fixtures/alphabeta_fixture.rs");

/// A deterministic stand-in for a random number generator, so a fuzz test is reproducible.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, bound: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % bound
    }
}

/// Unfinished positions from random play, which is enough variety for a differential test.
fn positions(count: usize, seed: u64) -> Vec<Connect4> {
    let mut rng = Lcg(seed);
    let mut found = Vec::with_capacity(count);
    while found.len() < count {
        let mut board = Connect4::new();
        let plies = 2 + rng.next(16);
        for _ in 0..plies {
            if board.is_game_over() {
                break;
            }
            let mut moves = [0u8; COLS as usize];
            let legal = board.legal_moves(&mut moves);
            board.make_move(moves[rng.next(legal)]);
        }
        if !board.is_game_over() {
            found.push(board);
        }
    }
    found
}

/// Negamax with no pruning at all: the answer alpha-beta has to reproduce.
///
/// Mirrors `search::negamax_ab` with the window taken out, including asking about terminality
/// before the horizon and including the depth term in the terminal score.
fn negamax(board: &mut Connect4, depth: i32, leaves: &mut u64) -> i32 {
    if let Some(outcome) = board.outcome() {
        return terminal_score(outcome, board.turn(), depth);
    }
    if depth == 0 {
        *leaves += 1;
        return weighted_eval(board);
    }

    let mut moves = [0u8; COLS as usize];
    let count = board.legal_moves(&mut moves);
    if count == 0 {
        return terminal_score(DRAW, board.turn(), depth);
    }

    let mut best = LOW_BOUND;
    for &column in moves[..count].iter() {
        board.make_move(column);
        let score = -negamax(board, depth - 1, leaves);
        board.unmake_move();
        if score > best {
            best = score;
        }
    }
    best
}

/// Every root move and the score the unpruned reference gives it, in generation order.
fn reference_scores(board: &mut Connect4, depth: i32) -> (Vec<(u8, i32)>, u64) {
    let mut moves = [0u8; COLS as usize];
    let count = board.legal_moves(&mut moves);
    let mut scored = Vec::with_capacity(count);
    let mut leaves = 0;

    for &column in moves[..count].iter() {
        board.make_move(column);
        scored.push((column, -negamax(board, depth - 1, &mut leaves)));
        board.unmake_move();
    }
    (scored, leaves)
}

fn scored(board: &mut Connect4, depth: i32) -> Vec<(u8, i32)> {
    let result = root_scores(board, depth);
    result.moves[..result.count].to_vec()
}

// ---- against the pinned fixture, without an interpreter -----------------------------------

#[test]
fn root_scores_match_the_pinned_python_answers() {
    for &(columns, depth, expected, expected_leaves) in FIXTURE {
        let mut board = Connect4::from_columns(columns);
        let result = root_scores(&mut board, depth);
        assert_eq!(
            &result.moves[..result.count],
            expected,
            "{columns:?} at depth {depth}"
        );
        assert_eq!(
            result.leaves, expected_leaves,
            "{columns:?} at depth {depth} walked a different tree"
        );
    }
}

// ---- the pruning is sound ----------------------------------------------------------------

#[test]
fn pruning_gives_the_same_score_to_every_root_move() {
    for mut board in positions(12, 20260814) {
        for depth in 1..=4 {
            let (expected, _) = reference_scores(&mut board.clone(), depth);
            assert_eq!(expected, scored(&mut board, depth), "at depth {depth}");
        }
    }
}

#[test]
fn the_two_searches_pick_the_same_move() {
    for mut board in positions(12, 1) {
        for depth in 1..=4 {
            let (reference, _) = reference_scores(&mut board.clone(), depth);
            let best = reference
                .iter()
                .fold((None, LOW_BOUND), |(best_move, best_score), &(column, score)| {
                    if score > best_score {
                        (Some(column), score)
                    } else {
                        (best_move, best_score)
                    }
                })
                .0;
            assert_eq!(best, best_move(&mut board, depth), "at depth {depth}");
        }
    }
}

#[test]
fn the_search_is_deterministic() {
    for mut board in positions(6, 2) {
        assert_eq!(scored(&mut board, 3), scored(&mut board, 3));
    }
}

#[test]
fn the_search_leaves_the_position_where_it_found_it() {
    for mut board in positions(8, 3) {
        let before = (board.discs(YELLOW), board.discs(RED), board.turn(), board.ply());
        best_move(&mut board, 4);
        assert_eq!(
            before,
            (board.discs(YELLOW), board.discs(RED), board.turn(), board.ply())
        );
    }
}

#[test]
fn ties_go_to_the_move_generated_first() {
    // Which for Connect 4 is the centre column, because generation is centre-first. With nothing
    // to choose between the moves the search takes the best opening move anyway.
    assert_eq!(Some((COLS / 2) as u8), best_move(&mut Connect4::new(), 2));
}

#[test]
fn alpha_beta_reaches_far_fewer_leaves_than_plain_negamax() {
    // From the empty board: 385 leaves against 2,401 at depth 4, and 923 against 16,807 at depth
    // 5. Asserted loosely on purpose - the exact counts are pinned against Python in
    // `tests/connect4/test_native.py`, where a difference means something rather than being a
    // number to re-edit whenever the evaluation changes what the search prefers.
    //
    // Weaker than the 24.8x in `test_search_equivalence.py`'s table because that measures one
    // search of the root, and this measures what `alpha_beta` does: a fresh full window per root
    // move, which discards everything a sibling had proved. It buys root independence, which
    // Connect 4 does not use and chess's `PARALLEL_ROOT` does.
    for depth in 4..=5 {
        let (_, unpruned) = reference_scores(&mut Connect4::new(), depth);
        let pruned = root_scores(&mut Connect4::new(), depth).leaves;

        assert_eq!(
            (COLS as u64).pow(depth as u32),
            unpruned,
            "the reference should search everything"
        );
        assert!(pruned * 5 < unpruned, "depth {depth}: {pruned} vs {unpruned}");
    }
}

// ---- scoring at and beside a finish -------------------------------------------------------

#[test]
fn a_win_in_one_scores_as_a_win() {
    let mut board = Connect4::from_columns(&[1, 0, 2, 0, 3, 0]);
    for (column, score) in scored(&mut board, 2) {
        if column == 4 {
            assert_eq!(MATE + 1, score, "column 4 completes the line");
        } else {
            assert!(score < MATE, "column {column} does not");
        }
    }
}

#[test]
fn a_closer_win_outscores_a_further_one() {
    // Both are won for Yellow and both are searched to the same depth. The only difference is how
    // long the win takes, and that has to show up in the score - otherwise the engine has no
    // reason to finish a game it has already won.
    let mut leaves = 0;
    let immediate = negamax_ab(
        &mut Connect4::from_columns(&[1, 0, 2, 0, 3, 0]),
        5,
        LOW_BOUND,
        HIGH_BOUND,
        &mut leaves,
    );
    // Yellow plays to make a three open at both ends, Red blocks one end, Yellow takes the other.
    let double_threat = negamax_ab(
        &mut Connect4::from_columns(&[2, 6, 3, 6]),
        5,
        LOW_BOUND,
        HIGH_BOUND,
        &mut leaves,
    );

    assert_eq!(MATE + 4, immediate, "won on the first of five plies");
    assert_eq!(MATE + 2, double_threat, "won on the third of five plies");
}

#[test]
fn a_loss_is_worth_the_same_from_either_side() {
    // Negamax reads every score from the point of view of whoever is to move, so a position one
    // player calls a win the other must call a loss of exactly the same size.
    let mut board = Connect4::from_columns(&[1, 0, 2, 0, 3, 0]);
    let mut leaves = 0;
    let winning = negamax(&mut board, 2, &mut leaves);

    board.make_move(4);
    assert_eq!(-winning, negamax(&mut board, 1, &mut leaves));
}

// ---- the evaluation ------------------------------------------------------------------------

#[test]
fn a_quiet_position_is_worth_exactly_nothing() {
    assert_eq!(0, value(&Connect4::new()));
    assert_eq!(0, value(&Connect4::from_columns(&[3, 3])));
}

#[test]
fn the_evaluation_reads_from_the_point_of_view_of_the_player_to_move() {
    for board in positions(200, 4) {
        let absolute = value(&board);
        let relative = weighted_eval(&board);
        assert_eq!(if board.turn() { absolute } else { -absolute }, relative);
    }
}

/// Or a merely good position becomes indistinguishable from a forced win, and the engine stops
/// trying to actually finish games. Checked at compile time because both sides are constants.
const _: () = assert!(MAX_EVAL < MATE / 10);

#[test]
fn the_evaluation_stays_inside_its_declared_budget() {
    for board in positions(2_000, 5) {
        let worth = value(&board);
        assert!(worth.abs() < MAX_EVAL, "{worth} is outside MAX_EVAL");
    }
}

#[test]
fn a_threat_is_worth_more_when_it_can_be_played() {
    // Three in a row along the bottom, completable now, against the same three floating a row up
    // behind a wall of the opponent's discs.
    let live = Connect4::from_columns(&[1, 0, 2, 0, 3, 0]);
    assert!(value(&live) > 0, "Yellow is one move from four");

    let empty = Connect4::new();
    assert_eq!(0, value(&empty));
}

// ---- crossing the boundary as two integers -------------------------------------------------

#[test]
fn a_position_survives_the_round_trip_through_its_bitboards() {
    for board in positions(200, 6) {
        let rebuilt = Connect4::from_discs(board.discs(YELLOW), board.discs(RED), board.turn())
            .expect("a played position is a reachable one");
        assert_eq!(board.discs(YELLOW), rebuilt.discs(YELLOW));
        assert_eq!(board.discs(RED), rebuilt.discs(RED));
        assert_eq!(board.turn(), rebuilt.turn());
        assert_eq!(board.occupied(), rebuilt.occupied());
    }
}

#[test]
fn the_search_gives_the_same_answer_to_a_rebuilt_position() {
    for mut board in positions(8, 7) {
        let mut rebuilt =
            Connect4::from_discs(board.discs(YELLOW), board.discs(RED), board.turn()).unwrap();
        assert_eq!(scored(&mut board, 4), scored(&mut rebuilt, 4));
    }
}

#[test]
fn an_unreachable_position_is_refused_rather_than_searched() {
    let bottom = 1u64;
    let floating = 1u64 << 2;

    assert!(Connect4::from_discs(bottom, bottom, RED).is_none(), "the same cell twice");
    assert!(
        Connect4::from_discs(1 << ROWS, 0, RED).is_none(),
        "a disc on column zero's sentinel"
    );
    assert!(
        Connect4::from_discs(!FULL_BOARD, 0, RED).is_none(),
        "discs off the board entirely"
    );
    assert!(
        Connect4::from_discs(floating, 0, RED).is_none(),
        "a disc floating above a gap"
    );
    assert!(
        Connect4::from_discs(bottom, 0, YELLOW).is_none(),
        "one disc played but Yellow still to move"
    );
    assert!(
        Connect4::from_discs(bottom, 1 << STRIDE, RED).is_none(),
        "level discs but Red to move"
    );

    assert!(Connect4::from_discs(0, 0, YELLOW).is_some(), "the empty board");
    assert!(Connect4::from_discs(bottom, 0, RED).is_some(), "one disc played");
}

#[test]
fn a_full_column_is_still_a_reachable_position() {
    let mut board = Connect4::new();
    for _ in 0..ROWS {
        board.make_move(0);
    }
    assert_eq!(COLUMN_MASKS[0], board.occupied());
    assert!(
        Connect4::from_discs(board.discs(YELLOW), board.discs(RED), board.turn()).is_some()
    );
}
