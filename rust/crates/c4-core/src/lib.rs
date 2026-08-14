//! Connect 4 self-play, fast enough to feed a GPU.
//!
//! A port of the parts of the Python engine a training run spends its time in - the board, the
//! encoder and the PUCT search - built to produce the *same* games rather than better ones. The
//! Python in `games/connect4/` and `ai/zero/` remains the readable definition; this is the same
//! thing at a speed that lets a generation be thousands of games instead of hundreds.

pub mod bitboard;
pub mod connect4;
pub mod constants;
pub mod encode;
pub mod mcts;
pub mod rng;
pub mod selfplay;

pub use connect4::{Connect4, Outcome, DRAW};
pub use mcts::{Search, SearchResult};
pub use rng::PyRandom;
pub use selfplay::{Config, SelfPlay};
