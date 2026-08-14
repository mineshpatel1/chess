//! The carry that drops a disc, and the shifts that find four in a row.
//!
//! A port of `games/connect4/bitboard.py`. Nothing here knows which player owns which cells: a
//! bitboard is one player's discs, or every occupied cell, depending on what the caller passes.

use crate::constants::{BOTTOM_ROW, COLUMN_MASKS, CONNECT, DIRECTIONS, FULL_BOARD, STRIDE};

/// The bit a cell occupies. Column-major, row 0 at the bottom.
pub const fn index(column: u32, row: u32) -> u32 {
    column * STRIDE + row
}

/// Where a disc would land in every column at once, one bit per non-full column.
///
/// The addition rings a carry up each column simultaneously and it settles on the first gap; a
/// full column's carry is absorbed by its sentinel and then cleared by the mask. This is why
/// there is no `height[]` array to keep in step.
pub fn drops(occupied: u64) -> u64 {
    (occupied + BOTTOM_ROW) & FULL_BOARD
}

/// Where a disc would land in one column, or zero if it is full.
pub fn landing_square(occupied: u64, column: u32) -> u64 {
    drops(occupied) & COLUMN_MASKS[column as usize]
}

/// The lowest cell of every run of `length` in one direction.
///
/// Doubling the shift and clamping to what is left means the total distance travelled is exactly
/// `length - 1`, so a run of four takes two ANDs rather than three. Nothing can wrap between
/// columns because the sentinel row is in the way.
pub fn runs(position: u64, delta: u32, length: u32) -> u64 {
    let mut mask = position;
    let mut remaining = length - 1;
    let mut step = 1;

    while remaining != 0 && mask != 0 {
        let shift = if step < remaining { step } else { remaining };
        mask &= mask >> (shift * delta);
        remaining -= shift;
        step *= 2;
    }
    mask
}

/// Whether these discs contain four in a row, in any direction.
pub fn is_win(position: u64) -> bool {
    DIRECTIONS
        .iter()
        .any(|&delta| runs(position, delta, CONNECT) != 0)
}
