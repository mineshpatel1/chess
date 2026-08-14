//! The board's shape, and every mask derived from it.
//!
//! A port of `games/connect4/constants.py`, and derived the same way it is: change `ROWS`, `COLS`
//! or `CONNECT` and everything below follows. Pasting the literals would work until the day
//! somebody wants a different board.

pub const ROWS: u32 = 6;
pub const COLS: u32 = 7;
pub const CONNECT: u32 = 4;

/// One *more* than the number of rows, so each column carries a sentinel cell above it that is
/// never occupied. That spare bit is what stops a shift chain wrapping between columns.
pub const STRIDE: u32 = ROWS + 1;

pub const YELLOW: bool = true;
pub const RED: bool = false;

pub const COLUMN_MASKS: [u64; COLS as usize] = column_masks();
pub const FULL_BOARD: u64 = or_all(&COLUMN_MASKS);
pub const BOTTOM_ROW: u64 = bottom_row();

pub const VERTICAL: u32 = 1;
pub const HORIZONTAL: u32 = STRIDE;
pub const DIAGONAL_UP: u32 = STRIDE + 1;
pub const DIAGONAL_DOWN: u32 = STRIDE - 1;
pub const DIRECTIONS: [u32; 4] = [VERTICAL, HORIZONTAL, DIAGONAL_UP, DIAGONAL_DOWN];

/// Columns nearest the middle first, which is worth a factor of nineteen at depth 6 and is also
/// the order children hang off a search node in.
pub const CENTRE_FIRST: [u8; COLS as usize] = centre_first();

const fn column_masks() -> [u64; COLS as usize] {
    let mut masks = [0u64; COLS as usize];
    let mut column = 0;
    while column < COLS as usize {
        masks[column] = ((1u64 << ROWS) - 1) << (column as u32 * STRIDE);
        column += 1;
    }
    masks
}

const fn or_all(masks: &[u64; COLS as usize]) -> u64 {
    let mut all = 0;
    let mut column = 0;
    while column < COLS as usize {
        all |= masks[column];
        column += 1;
    }
    all
}

const fn bottom_row() -> u64 {
    let mut row = 0;
    let mut column = 0;
    while column < COLS {
        row |= 1u64 << (column * STRIDE);
        column += 1;
    }
    row
}

/// Sorted by distance from the centre, ties to the lower column - an insertion sort because a
/// `const fn` cannot call `sort_by_key`.
const fn centre_first() -> [u8; COLS as usize] {
    let mut order = [0u8; COLS as usize];
    let mut column = 0;
    while column < COLS as usize {
        let key = distance_from_centre(column as u32);
        let mut position = column;
        while position > 0 && distance_from_centre(order[position - 1] as u32) > key {
            order[position] = order[position - 1];
            position -= 1;
        }
        order[position] = column as u8;
        column += 1;
    }
    order
}

const fn distance_from_centre(column: u32) -> u32 {
    (2 * column).abs_diff(COLS - 1)
}
