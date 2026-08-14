//! How good a Connect 4 position is, in the terms the search wants it.
//!
//! A port of `games/connect4/evaluation.py`, where the argument for it lives: what is counted is
//! open threes and nothing else, and a quiet position scores exactly zero on purpose - that being
//! the policy "take the centre unless there is a tactic", which the search's tie-break supplies
//! for free. Three further terms were built, measured and deleted there. The numbers are in that
//! docstring; this is not the place to try them again.

use crate::bitboard::{drops, runs};
use crate::connect4::Connect4;
use crate::constants::{CONNECT, DIRECTIONS, FULL_BOARD, RED, VERTICAL, YELLOW};

/// Every score has to stay well inside `search::MATE`, or a merely good position becomes
/// indistinguishable from a forced win and the engine stops trying to finish games.
pub const MAX_EVAL: i32 = 10_000;

/// Per cell that would complete a run of three into a win.
const THREE: i32 = 5;

/// Per cell, by whether it can be played this move or is floating above the stack.
const PLAYABLE: i32 = 3;
const DISTANT: i32 = 1;

/// Per cell, by direction. A vertical threat is blocked by playing on top of it, for free.
const UPRIGHT: i32 = 1;
const ACROSS: i32 = 3;

/// `(direction, value of a playable completion cell, value of a distant one)`, multiplied out
/// once so the leaf does one lookup and two multiplies per direction.
pub const THREAT_TERMS: [(u32, i32, i32); DIRECTIONS.len()] = threat_terms();

/// Runs one short of a win. Named rather than inlined because `threat_cells` is written for any
/// length, and the choice to only ever ask it for this one is a result rather than a detail.
pub const THREAT_LENGTH: u32 = CONNECT - 1;

const fn threat_terms() -> [(u32, i32, i32); DIRECTIONS.len()] {
    let mut terms = [(0u32, 0i32, 0i32); DIRECTIONS.len()];
    let mut slot = 0;
    while slot < DIRECTIONS.len() {
        let delta = DIRECTIONS[slot];
        let weight = if delta == VERTICAL { UPRIGHT } else { ACROSS };
        terms[slot] = (delta, THREE * weight * PLAYABLE, THREE * weight * DISTANT);
        slot += 1;
    }
    terms
}

/// The empty cells that would extend a run of `length` in `position` by one.
///
/// `runs` marks the lowest cell of every run, so the two cells that would extend it are one step
/// below that mark and `length` steps above it. Intersecting with the empty cells drops the ends
/// that are occupied, the ones that walked off the board and the ones that landed on a sentinel,
/// all at once - which is also why the left shift may carry bits past 63 without harm.
pub fn threat_cells(position: u64, delta: u32, length: u32, empty: u64) -> u64 {
    let anchors = runs(position, delta, length);
    if anchors == 0 {
        return 0;
    }
    ((anchors << (length * delta)) | (anchors >> delta)) & empty
}

/// One player's open threes, weighted by direction and by whether they are live.
fn threat_value(position: u64, empty: u64, playable: u64) -> i32 {
    let mut total = 0;
    for &(delta, playable_weight, distant_weight) in THREAT_TERMS.iter() {
        let cells = threat_cells(position, delta, THREAT_LENGTH, empty);
        if cells == 0 {
            continue;
        }
        total += playable_weight * (cells & playable).count_ones() as i32;
        total += distant_weight * (cells & !playable).count_ones() as i32;
    }
    total
}

/// How much better Yellow's position is than Red's. Positive favours Yellow.
///
/// Says nothing about a position that is already won; that is the search's business, and
/// `outcome` is checked before this is reached.
pub fn value(board: &Connect4) -> i32 {
    let yellow = board.discs(YELLOW);
    let red = board.discs(RED);
    let occupied = yellow | red;
    let empty = FULL_BOARD & !occupied;
    let playable = drops(occupied);

    threat_value(yellow, empty, playable) - threat_value(red, empty, playable)
}

/// The evaluation the search uses, read from the point of view of the player to move.
///
/// The sign flip lives here and nowhere else. Getting it wrong gives an engine that plays well at
/// even depths and badly at odd ones, which is hard to notice and easy to avoid.
pub fn weighted_eval(board: &Connect4) -> i32 {
    if board.turn() {
        value(board)
    } else {
        -value(board)
    }
}
