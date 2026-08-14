//! `Connect4`: two integers, move generation, make/unmake, win detection.
//!
//! A port of `games/connect4/board.py`, holding only what a search needs - the evaluation, the
//! solver keys and the diagram parsing stay in Python.

use crate::bitboard::{drops, is_win, landing_square};
use crate::constants::{CENTRE_FIRST, COLS, COLUMN_MASKS, RED, YELLOW};

/// How a finished game finished. `None` for a winner means a draw, as `games/base.py` has it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Outcome {
    pub winner: Option<bool>,
}

pub const DRAW: Outcome = Outcome { winner: None };

/// Two bitboards rather than a position/mask pair: make and unmake become the same XOR, and it is
/// one player's discs an evaluation wants to look at.
#[derive(Clone, Debug)]
pub struct Connect4 {
    discs: [u64; 2],
    turn: bool,
    move_stack: Vec<u64>,
}

impl Default for Connect4 {
    fn default() -> Self {
        Self::new()
    }
}

impl Connect4 {
    pub fn new() -> Self {
        Connect4 {
            discs: [0; 2],
            turn: YELLOW,
            move_stack: Vec::with_capacity((COLS * crate::constants::ROWS) as usize),
        }
    }

    /// Replays a sequence of columns, panicking on an illegal one.
    pub fn from_columns(columns: &[u8]) -> Self {
        let mut board = Self::new();
        for &column in columns {
            assert!(board.is_legal(column), "column {column} is not playable");
            board.make_move(column);
        }
        board
    }

    pub fn turn(&self) -> bool {
        self.turn
    }

    pub fn discs(&self, player: bool) -> u64 {
        self.discs[player as usize]
    }

    pub fn ply(&self) -> usize {
        self.move_stack.len()
    }

    /// Derived rather than cached: a cached copy is one more thing make and unmake must keep true.
    pub fn occupied(&self) -> u64 {
        self.discs[YELLOW as usize] | self.discs[RED as usize]
    }

    /// The playable columns, centre first, written into `out`; returns how many there are.
    pub fn legal_moves(&self, out: &mut [u8; COLS as usize]) -> usize {
        let landing = drops(self.occupied());
        let mut count = 0;
        for &column in CENTRE_FIRST.iter() {
            if landing & COLUMN_MASKS[column as usize] != 0 {
                out[count] = column;
                count += 1;
            }
        }
        count
    }

    pub fn is_legal(&self, column: u8) -> bool {
        landing_square(self.occupied(), column as u32) != 0
    }

    pub fn has_moves(&self) -> bool {
        drops(self.occupied()) != 0
    }

    /// Assumes the move is legal; establishing that is `legal_moves`' job.
    pub fn make_move(&mut self, column: u8) {
        let bit = landing_square(self.occupied(), column as u32);
        debug_assert!(bit != 0, "column {column} is full");
        self.discs[self.turn as usize] ^= bit;
        self.move_stack.push(bit);
        self.turn = !self.turn;
    }

    pub fn unmake_move(&mut self) {
        let bit = self.move_stack.pop().expect("no move to unmake");
        self.turn = !self.turn;
        self.discs[self.turn as usize] ^= bit;
    }

    /// A win, if there is one. Only the player who just moved can have just won, so only their
    /// board is tested.
    pub fn outcome(&self) -> Option<Outcome> {
        let mover = !self.turn;
        if is_win(self.discs[mover as usize]) {
            Some(Outcome {
                winner: Some(mover),
            })
        } else {
            None
        }
    }

    /// How the game finished, or `None` if it has not.
    pub fn result(&self) -> Option<Outcome> {
        match self.outcome() {
            Some(outcome) => Some(outcome),
            None if !self.has_moves() => Some(DRAW),
            None => None,
        }
    }

    pub fn is_game_over(&self) -> bool {
        self.result().is_some()
    }

    /// The value of a finished position to the player whose turn it would be: the convention the
    /// training targets and `ai/search.py` also use.
    pub fn terminal_value(&self) -> f64 {
        match self.result().expect("not a finished position").winner {
            None => 0.0,
            Some(winner) => {
                if winner == self.turn {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }
}
