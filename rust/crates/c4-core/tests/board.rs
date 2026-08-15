//! The board against the counts and invariants `tests/connect4/` pins for the Python engine.

use c4_core::bitboard::{drops, index, is_win, landing_square, runs};
use c4_core::constants::{
    BOTTOM_ROW, CENTRE_FIRST, COLS, COLUMN_MASKS, CONNECT, DIRECTIONS, FULL_BOARD, ROWS, STRIDE,
};
use c4_core::encode::{legal_mask, planes, PLANE_BYTES, PLANE_SIZE};
use c4_core::Connect4;

/// A deterministic stand-in for a random number generator, so a fuzz test is reproducible.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, bound: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % bound
    }
}

// ---- the masks, re-derived independently ------------------------------------------------

#[test]
fn the_board_is_every_playable_cell_and_nothing_else() {
    let mut expected = 0u64;
    for column in 0..COLS {
        for row in 0..ROWS {
            expected |= 1 << index(column, row);
        }
    }
    assert_eq!(FULL_BOARD, expected);
    assert_eq!(FULL_BOARD.count_ones(), ROWS * COLS);
}

#[test]
fn the_sentinel_row_is_never_part_of_the_board() {
    for column in 0..COLS {
        assert_eq!(FULL_BOARD & (1 << index(column, ROWS)), 0);
    }
}

#[test]
fn column_masks_tile_the_board() {
    let mut union = 0u64;
    for mask in COLUMN_MASKS {
        assert_eq!(union & mask, 0, "columns overlap");
        union |= mask;
    }
    assert_eq!(union, FULL_BOARD);
}

#[test]
fn the_bottom_row_is_one_cell_per_column() {
    assert_eq!(BOTTOM_ROW.count_ones(), COLS);
    for column in 0..COLS {
        assert_eq!(BOTTOM_ROW & COLUMN_MASKS[column as usize], 1 << index(column, 0));
    }
}

#[test]
fn moves_are_offered_centre_first() {
    assert_eq!(CENTRE_FIRST, [3, 2, 4, 1, 5, 0, 6]);
}

// ---- win detection ----------------------------------------------------------------------

/// Four in a row found by scanning the grid, sharing nothing with the shift chains.
fn naive_is_win(position: u64) -> bool {
    let steps: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
    for column in 0..COLS as i32 {
        for row in 0..ROWS as i32 {
            for (dc, dr) in steps {
                let mut found = 0;
                for step in 0..CONNECT as i32 {
                    let (c, r) = (column + dc * step, row + dr * step);
                    if c < 0 || c >= COLS as i32 || r < 0 || r >= ROWS as i32 {
                        break;
                    }
                    if position >> index(c as u32, r as u32) & 1 == 0 {
                        break;
                    }
                    found += 1;
                }
                if found == CONNECT as i32 {
                    return true;
                }
            }
        }
    }
    false
}

#[test]
fn every_line_of_four_is_a_win() {
    let steps: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
    let mut lines = 0;
    for column in 0..COLS as i32 {
        for row in 0..ROWS as i32 {
            for (dc, dr) in steps {
                let cells: Vec<(i32, i32)> = (0..CONNECT as i32)
                    .map(|step| (column + dc * step, row + dr * step))
                    .collect();
                if cells.iter().any(|&(c, r)| {
                    c < 0 || c >= COLS as i32 || r < 0 || r >= ROWS as i32
                }) {
                    continue;
                }
                let position = cells
                    .iter()
                    .fold(0u64, |bb, &(c, r)| bb | 1 << index(c as u32, r as u32));
                assert!(is_win(position), "missed a line at {column},{row} {dc},{dr}");
                lines += 1;
            }
        }
    }
    assert_eq!(lines, 69, "the 7x6 board has 69 lines of four");
}

#[test]
fn a_run_cannot_wrap_between_columns() {
    // Four cells that are consecutive in the *bit* layout but span two columns in every direction.
    for delta in DIRECTIONS {
        for start in 0..(COLS * STRIDE - CONNECT * delta) {
            let position = (0..CONNECT).fold(0u64, |bb, step| bb | 1 << (start + step * delta));
            let inside = position & FULL_BOARD == position;
            assert_eq!(
                is_win(position & FULL_BOARD),
                inside && naive_is_win(position & FULL_BOARD),
                "delta {delta} from bit {start}"
            );
        }
    }
}

#[test]
fn win_detection_agrees_with_a_grid_scan_over_random_boards() {
    let mut rng = Lcg(20260814);
    for _ in 0..20_000 {
        let mut board = Connect4::new();
        let mut moves = [0u8; COLS as usize];
        loop {
            let count = board.legal_moves(&mut moves);
            if count == 0 || board.outcome().is_some() {
                break;
            }
            board.make_move(moves[rng.next(count)]);
        }
        for player in [true, false] {
            assert_eq!(is_win(board.discs(player)), naive_is_win(board.discs(player)));
        }
    }
}

#[test]
fn runs_travels_exactly_length_minus_one() {
    // A vertical line of three has its lowest cell marked at length 3 and nothing at length 4.
    let three = 1 | 1 << 1 | 1 << 2;
    assert_eq!(runs(three, 1, 3), 1);
    assert_eq!(runs(three, 1, 4), 0);
    assert_eq!(runs(three | 1 << 3, 1, 4), 1);
    assert_eq!(runs(three, 1, 2), 1 | 1 << 1);
}

// ---- move generation and make/unmake -----------------------------------------------------

#[test]
fn a_carry_finds_the_landing_square_in_every_column() {
    let mut board = Connect4::new();
    for column in 0..COLS {
        for row in 0..ROWS {
            assert_eq!(landing_square(board.occupied(), column), 1 << index(column, row));
            board.make_move(column as u8);

            // A move somewhere else, taken straight back: unmaking must leave this column's
            // landing square where it was. There is nowhere spare once the board is nearly full,
            // and the assertion above is the point of the loop either way.
            let spare = (0..COLS)
                .find(|&other| other != column && landing_square(board.occupied(), other) != 0);
            if let Some(other) = spare {
                board.make_move(other as u8);
                board.unmake_move();
            }
        }
        assert_eq!(landing_square(board.occupied(), column), 0, "full column still offered");
    }
    assert_eq!(drops(board.occupied()), 0);
    assert!(!board.has_moves());
}

#[test]
fn column_zero_alone_is_still_a_move() {
    // The trap `games/base.py` documents: move 0 is legal and falsy in Python.
    let mut board = Connect4::new();
    for column in 1..COLS {
        for _ in 0..ROWS {
            board.make_move(column as u8);
        }
    }
    let mut moves = [0u8; COLS as usize];
    assert_eq!(board.legal_moves(&mut moves), 1);
    assert_eq!(moves[0], 0);
    assert!(board.has_moves());
    assert!(!board.is_game_over() || board.outcome().is_some());
}

#[test]
fn unmake_restores_the_position_exactly() {
    let mut rng = Lcg(1);
    let mut board = Connect4::new();
    let mut moves = [0u8; COLS as usize];

    for _ in 0..50_000 {
        let count = board.legal_moves(&mut moves);
        if count == 0 || board.outcome().is_some() {
            board = Connect4::new();
            continue;
        }
        let before = (board.discs(true), board.discs(false), board.turn(), board.ply());
        let column = moves[rng.next(count)];

        board.make_move(column);
        board.unmake_move();
        assert_eq!(
            (board.discs(true), board.discs(false), board.turn(), board.ply()),
            before
        );
        board.make_move(column);
    }
}

// ---- perft -------------------------------------------------------------------------------

/// Leaves below this position, counting a finished game as having none. The horizon is checked
/// before the outcome, so a win at exactly `depth` counts as one leaf - see `ai/perft.py`.
fn perft(board: &mut Connect4, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if board.outcome().is_some() {
        return 0;
    }

    let mut moves = [0u8; COLS as usize];
    let count = board.legal_moves(&mut moves);
    let mut total = 0;
    for &column in moves.iter().take(count) {
        board.make_move(column);
        total += perft(board, depth - 1);
        board.unmake_move();
    }
    total
}

#[test]
fn perft_matches_the_pinned_counts() {
    let expected = [7, 49, 343, 2_401, 16_807, 117_649, 823_536];
    let mut board = Connect4::new();
    for (depth, &nodes) in expected.iter().enumerate() {
        assert_eq!(perft(&mut board, depth as u32 + 1), nodes, "depth {}", depth + 1);
    }
}

#[test]
#[ignore = "about a second; the shorter depths catch the same faults"]
fn perft_eight_matches_the_pinned_count() {
    assert_eq!(perft(&mut Connect4::new(), 8), 5_673_234);
}

// ---- the encoder --------------------------------------------------------------------------

#[test]
fn plane_zero_is_always_the_mover() {
    let mut board = Connect4::from_columns(&[3, 3, 4]);
    let mut mine = [0u8; PLANE_BYTES];
    planes(&board, &mut mine);

    // Yellow has two discs on the board and is to move again after four plies.
    assert_eq!(mine[..PLANE_SIZE].iter().map(|&c| c as u32).sum::<u32>(), 1);
    assert_eq!(mine[PLANE_SIZE..].iter().map(|&c| c as u32).sum::<u32>(), 2);

    board.make_move(0);
    let mut theirs = [0u8; PLANE_BYTES];
    planes(&board, &mut theirs);
    assert_eq!(theirs[..PLANE_SIZE], mine[PLANE_SIZE..]);
}

#[test]
fn row_zero_is_the_bottom_of_the_board() {
    let board = Connect4::from_columns(&[6]);
    let mut cells = [0u8; PLANE_BYTES];
    planes(&board, &mut cells);

    // The mover is now Red, so Yellow's lone disc is in plane 1, bottom row, column 6.
    assert_eq!(cells[PLANE_SIZE + (COLS - 1) as usize], 1);
    assert_eq!(cells.iter().map(|&c| c as u32).sum::<u32>(), 1);
}

#[test]
fn the_legal_mask_is_the_legal_moves() {
    let mut rng = Lcg(7);
    let mut board = Connect4::new();
    let mut moves = [0u8; COLS as usize];

    for _ in 0..5_000 {
        let count = board.legal_moves(&mut moves);
        if count == 0 || board.outcome().is_some() {
            board = Connect4::new();
            continue;
        }
        let mut mask = [false; COLS as usize];
        legal_mask(&board, &mut mask);
        assert_eq!(mask.iter().filter(|&&legal| legal).count(), count);
        for &column in moves.iter().take(count) {
            assert!(mask[column as usize]);
        }
        board.make_move(moves[rng.next(count)]);
    }
}
