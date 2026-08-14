//! Connect 4, fast enough to feed a GPU and to measure what it learns.
//!
//! A port of the parts of the Python engine a training run spends its time in - the board, the
//! encoder, the PUCT search that plays a generation and the alpha-beta that grades it - built to
//! produce the *same* games and the same answers rather than better ones. The Python in
//! `games/connect4/`, `ai/zero/` and `ai/search.py` remains the readable definition; this is the
//! same thing at a speed that lets a generation be thousands of games instead of hundreds, and a
//! ladder rung minutes instead of hours.

pub mod bitboard;
pub mod connect4;
pub mod constants;
pub mod encode;
pub mod evaluation;
pub mod mcts;
pub mod rng;
pub mod search;
pub mod selfplay;

pub use connect4::{Connect4, Outcome, DRAW};
pub use mcts::{Search, SearchResult};
pub use rng::PyRandom;
pub use search::RootScores;
pub use selfplay::{Config, SelfPlay};
