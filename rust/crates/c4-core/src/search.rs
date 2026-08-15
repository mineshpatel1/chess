//! Fixed-depth alpha-beta, in the negamax formulation.
//!
//! A port of `ai/search.py`, operation for operation: the same fail-hard window, the same full
//! window per root move, the same tie to the move generated first. It is the ladder's opponent,
//! and the only reason it is here is that `minimax:8` costs 391ms a move in Python.
//!
//! No transposition table, no killers, no ordering beyond `CENTRE_FIRST` - because the Python has
//! none of those either. Each would be worth real speed and each would change which move comes
//! back, and a change of player cannot be checked by an equality test. This one can.

use crate::connect4::{Connect4, Outcome, DRAW};
use crate::constants::COLS;
use crate::evaluation::weighted_eval;

pub const LOW_BOUND: i32 = -9_999_999;
pub const HIGH_BOUND: i32 = 9_999_999;

/// Score of a forced win, kept well inside the search window so it never collides with the
/// alpha/beta bounds themselves.
pub const MATE: i32 = 1_000_000;

/// Every root move and what a full-window search made of it.
pub struct RootScores {
    /// `(column, score)` in generation order, of which the first `count` are filled.
    pub moves: [(u8, i32); COLS as usize],
    pub count: usize,
    /// Leaves reached, which is how often the evaluation was called. Not needed to play, but it
    /// is what says the Rust search walked the *same tree* as the Python rather than merely
    /// arriving at the same answer.
    pub leaves: u64,
}

impl RootScores {
    /// The move `alpha_beta` picks from these: highest score, ties to the first generated.
    pub fn best(&self) -> Option<u8> {
        let mut best_move = None;
        let mut best_score = LOW_BOUND;
        for &(column, score) in self.moves[..self.count].iter() {
            if score > best_score {
                best_score = score;
                best_move = Some(column);
            }
        }
        best_move
    }
}

/// Scores a finished position from the point of view of the player to move in it.
///
/// Deeper wins score lower, `depth` being the depth still remaining, so a mate in one is
/// preferred to a mate in five and the engine does not shuffle in a won position.
pub fn terminal_score(outcome: Outcome, turn: bool, depth: i32) -> i32 {
    match outcome.winner {
        None => 0,
        Some(winner) => {
            if winner == turn {
                MATE + depth
            } else {
                -(MATE + depth)
            }
        }
    }
}

/// Alpha-beta in the negamax formulation: every node returns a score from the point of view of
/// the player to move in it, and its parent negates what it gets back.
///
/// Terminality is asked about *before* the horizon, so a game already won scores as a win rather
/// than being handed to the evaluation. Connect 4 needs that - it is won with the board half
/// empty and legal moves still on offer.
///
/// Public because it is what the tests compare against an unpruned reference, exactly as
/// `tests/connect4/test_search_equivalence.py` reaches for `ai.search._negamax_ab`. Playing goes
/// through `best_move`.
pub fn negamax_ab(
    board: &mut Connect4,
    depth: i32,
    mut alpha: i32,
    beta: i32,
    leaves: &mut u64,
) -> i32 {
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
        // Nothing to play means the grid is full, and a full grid nobody won is a draw. This is
        // `outcome_without_moves` in Python, which for Connect 4 is always a draw.
        return terminal_score(DRAW, board.turn(), depth);
    }

    for &column in moves[..count].iter() {
        board.make_move(column);
        let score = -negamax_ab(board, depth - 1, -beta, -alpha, leaves);
        board.unmake_move();

        if score >= beta {
            return beta; // Fail-hard
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

/// Scores every root move on a full window.
///
/// A full window per root move is weaker pruning than carrying alpha across them, but it is what
/// makes each score the exact minimax value of that move rather than a bound - which is what lets
/// the Python and Rust searches be compared element for element rather than on the answer alone.
pub fn root_scores(board: &mut Connect4, depth: i32) -> RootScores {
    assert!(depth > 0, "a search has to look at least one ply ahead");

    let mut moves = [0u8; COLS as usize];
    let count = board.legal_moves(&mut moves);
    let mut scored = [(0u8, 0i32); COLS as usize];
    let mut leaves = 0;

    for slot in 0..count {
        let column = moves[slot];
        board.make_move(column);
        let score = -negamax_ab(board, depth - 1, LOW_BOUND, HIGH_BOUND, &mut leaves);
        board.unmake_move();
        scored[slot] = (column, score);
    }

    RootScores {
        moves: scored,
        count,
        leaves,
    }
}

/// The move a `depth`-ply search picks, or `None` on a board with nowhere left to play.
pub fn best_move(board: &mut Connect4, depth: i32) -> Option<u8> {
    root_scores(board, depth).best()
}
