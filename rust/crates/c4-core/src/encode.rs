//! What a network sees: two planes of 6x7, seven shared actions.
//!
//! A port of `games/connect4/encoding.py`. Plane 0 is the mover's discs and plane 1 the
//! opponent's, so a position and its colour-swap produce the same tensor; row 0 is the bottom of
//! the board.

use crate::bitboard::index;
use crate::connect4::Connect4;
use crate::constants::{COLS, COLUMN_MASKS, ROWS};

pub const PLANES: usize = 2;
pub const PLANE_SIZE: usize = (ROWS * COLS) as usize;
pub const PLANE_BYTES: usize = PLANES * PLANE_SIZE;
pub const POLICY_SIZE: usize = COLS as usize;

/// Writes the two planes into `out`, laid out as `[plane][row][column]`.
pub fn planes(state: &Connect4, out: &mut [u8]) {
    debug_assert_eq!(out.len(), PLANE_BYTES);
    let boards = [state.discs(state.turn()), state.discs(!state.turn())];

    for (plane, board) in boards.iter().enumerate() {
        for row in 0..ROWS {
            for column in 0..COLS {
                let cell = plane * PLANE_SIZE + (row * COLS + column) as usize;
                out[cell] = ((board >> index(column, row)) & 1) as u8;
            }
        }
    }
}

/// Which actions are legal, in action-space order rather than the order moves are generated in.
pub fn legal_mask(state: &Connect4, out: &mut [bool; POLICY_SIZE]) {
    let landing = crate::bitboard::drops(state.occupied());
    for column in 0..COLS as usize {
        out[column] = landing & COLUMN_MASKS[column] != 0;
    }
}
