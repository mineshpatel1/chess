//! The Rust self-play engine, as a Python extension module.
//!
//! The contract is the one `ai/zero/selfplay.py` already works to: the search hands out the
//! positions it needs evaluated and is handed back the answers. What changes is which side of the
//! boundary each half lives on. Rust owns the games, the trees and the encoding; PyTorch keeps
//! owning the network, so there is no second copy of the model, nothing to export between
//! generations, and no numerical parity to argue about.
//!
//! ```python
//! sp = zero_rs.SelfPlay(games=2000, simulations=600, exploration=2.0, seed='1')
//! while (batch := sp.pending()) is not None:
//!     planes, legal = batch
//!     logits, value = net(to_tensor(planes))
//!     sp.submit(softmax(logits.masked_fill(~legal, MASKED)), value)
//! planes, policy, value = sp.examples()
//! ```
//!
//! Alongside it, `best_move`/`root_scores`/`evaluate` are a second port with nothing to do with
//! the network: a fixed-depth alpha-beta, for the ladder's Python opponents rather than the
//! challenger. A position crosses as two bitboards and a turn, which is what a live board already
//! holds - there is no reason to hand across a column list and replay it.

use c4_core::constants::{COLS, ROWS};
use c4_core::encode::{PLANE_BYTES, POLICY_SIZE};
use c4_core::evaluation::value as evaluation_value;
use c4_core::mcts::State as SearchState;
use c4_core::search::{best_move as search_best_move, root_scores as search_root_scores, MATE};
use c4_core::selfplay::{Config, SelfPlay as Driver};
use c4_core::{Connect4, PyRandom, Search};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray4, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The planes and the legality mask for one batch of positions.
type Batch<'py> = (Bound<'py, PyArray4<u8>>, Bound<'py, PyArray2<bool>>);

/// A generation's training examples: planes, policy targets, value targets.
type Examples<'py> = (
    Bound<'py, PyArray4<i8>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
);

/// A position's planes as nested lists, and which of its actions are legal.
type Encoded = (Vec<Vec<Vec<i64>>>, Vec<bool>);

/// A generation of Connect 4 self-play games, advanced together.
#[pyclass(module = "zero_rs")]
struct SelfPlay {
    driver: Driver,
    games: usize,
}

#[pymethods]
impl SelfPlay {
    #[new]
    #[pyo3(signature = (
        games,
        simulations,
        exploration,
        seed,
        dirichlet_alpha = 1.0,
        dirichlet_epsilon = 0.25,
        temperature = 1.0,
        final_temperature = 0.0,
        temperature_moves = 30,
        opening_plies = 0,
        in_flight = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        games: usize,
        simulations: u32,
        exploration: f64,
        seed: String,
        dirichlet_alpha: f64,
        dirichlet_epsilon: f64,
        temperature: f64,
        final_temperature: f64,
        temperature_moves: usize,
        opening_plies: u32,
        in_flight: Option<usize>,
    ) -> Self {
        // Every game in flight by default: the batch is what the GPU is here for, and the trees
        // cost little enough that holding all of them is not worth economising on.
        let in_flight = in_flight.unwrap_or(games);
        SelfPlay {
            driver: Driver::new(Config {
                games,
                in_flight,
                simulations,
                exploration,
                dirichlet_alpha,
                dirichlet_epsilon,
                temperature,
                final_temperature,
                temperature_moves,
                opening_plies,
                seed,
            }),
            games,
        }
    }

    /// The positions every game in flight is waiting on, or `None` once they have all finished.
    ///
    /// Returns the planes as `uint8` of shape `(batch, 2, 6, 7)` and a legality mask as `bool` of
    /// shape `(batch, 7)`. Both must be answered by `submit` before this is called again.
    fn pending<'py>(&mut self, py: Python<'py>) -> Option<Batch<'py>> {
        let batch = self.driver.pending()?;
        let positions = batch.positions;

        let planes = PyArray1::from_slice(py, batch.planes)
            .reshape([positions, 2, ROWS as usize, COLS as usize])
            .expect("the planes buffer is one batch of boards");
        let legal = PyArray1::from_slice(py, batch.legal)
            .reshape([positions, POLICY_SIZE])
            .expect("the mask buffer is one batch of actions");
        Some((planes, legal))
    }

    /// Answers the batch `pending` reported: a policy over the whole action space per position,
    /// and one value per position, both from the mover's point of view.
    fn submit(
        &mut self,
        py: Python<'_>,
        priors: PyReadonlyArray2<'_, f32>,
        values: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let priors = priors
            .as_slice()
            .map_err(|_| PyValueError::new_err("priors must be contiguous"))?;
        let values = values
            .as_slice()
            .map_err(|_| PyValueError::new_err("values must be contiguous"))?;

        // The GIL is not needed for any of this, and holding it would stop anything else in the
        // process running while a generation's worth of tree work goes by.
        py.detach(|| self.driver.submit(priors, values));
        Ok(())
    }

    /// Every finished game's examples, in game order: planes, policy targets, value targets.
    ///
    /// The three-array shape is the one `ai/zero/replay.py` already writes, so nothing has to be
    /// taken apart and put back together on the way to the buffer.
    fn examples<'py>(&self, py: Python<'py>) -> PyResult<Examples<'py>> {
        let positions = self.driver.positions();
        let mut planes = Vec::with_capacity(positions * PLANE_BYTES);
        let mut policy = Vec::with_capacity(positions * POLICY_SIZE);
        let mut value = Vec::with_capacity(positions);

        for record in self.driver.records() {
            planes.extend(record.planes.iter().map(|&cell| cell as i8));
            policy.extend(record.policies.iter().map(|&share| share as f32));
            value.extend(record.values.iter().map(|&outcome| outcome as f32));
        }

        Ok((
            planes
                .into_pyarray(py)
                .reshape([positions, 2, ROWS as usize, COLS as usize])?,
            policy.into_pyarray(py).reshape([positions, POLICY_SIZE])?,
            value.into_pyarray(py),
        ))
    }

    /// How many examples each game produced, in game order, which is also how long it ran.
    fn lengths(&self) -> Vec<usize> {
        self.driver.records().map(|record| record.positions).collect()
    }

    /// How many games ended in a draw.
    fn drawn(&self) -> usize {
        self.driver
            .records()
            .filter(|record| record.winner.is_none())
            .count()
    }

    #[getter]
    fn completed(&self) -> usize {
        self.driver.completed()
    }

    #[getter]
    fn games(&self) -> usize {
        self.games
    }
}

// ---- hooks for tests/zero/test_fast.py ------------------------------------------------------
//
// Both of these exist so the Python suite can compare this engine against `ai/zero/` directly
// rather than against numbers copied out of it once. `search` takes the evaluator as a callable
// so that the two implementations are handed the *same* Python function, which leaves nothing
// about the comparison for a second implementation of it to get wrong.

fn board_from(columns: &[u8]) -> PyResult<Connect4> {
    let mut board = Connect4::new();
    for &column in columns {
        if column as u32 >= COLS || !board.is_legal(column) {
            return Err(PyValueError::new_err(format!("column {column} is not playable")));
        }
        board.make_move(column);
    }
    Ok(board)
}

// ---- alpha-beta: the ladder's opponent -------------------------------------------------------
//
// A position crosses the boundary as two integers rather than a column list, because the ladder
// hands this a live Python board move by move and has no reason to replay it from the start.
// `Connect4::from_discs` is where an impossible position is refused.

fn board_from_discs(yellow: u64, red: u64, turn: bool) -> PyResult<Connect4> {
    Connect4::from_discs(yellow, red, turn)
        .ok_or_else(|| PyValueError::new_err("yellow and red do not describe a reachable position"))
}

/// The move a `depth`-ply alpha-beta search picks, given the position as two bitboards.
///
/// A fixed-depth port of `ai.search.alpha_beta` with `games.connect4.evaluation.weighted_eval` -
/// same fail-hard window, same tie to the move generated first, no transposition table.
#[pyfunction]
fn best_move(py: Python<'_>, yellow: u64, red: u64, turn: bool, depth: i32) -> PyResult<u8> {
    let mut board = board_from_discs(yellow, red, turn)?;
    // The GIL buys nothing here and a slow rung should not hold up anything else in the process.
    py.detach(|| search_best_move(&mut board, depth))
        .ok_or_else(|| PyValueError::new_err("no legal moves"))
}

/// Every legal move and its full-window minimax score, in generation order - the test hook that
/// lets `tests/connect4/test_native.py` compare element for element rather than on the move alone.
#[pyfunction]
fn root_scores(py: Python<'_>, yellow: u64, red: u64, turn: bool, depth: i32) -> PyResult<Vec<(u8, i32)>> {
    let mut board = board_from_discs(yellow, red, turn)?;
    let result = py.detach(|| search_root_scores(&mut board, depth));
    Ok(result.moves[..result.count].to_vec())
}

/// The static evaluation alone, from Yellow's point of view - `games.connect4.evaluation.value`.
///
/// `value` itself does not care whose turn it is, but validating that `yellow` and `red` describe
/// a reachable position does, so this still takes `turn` and passes it straight to `from_discs`.
#[pyfunction]
fn evaluate(yellow: u64, red: u64, turn: bool) -> PyResult<i32> {
    let board = board_from_discs(yellow, red, turn)?;
    Ok(evaluation_value(&board))
}

/// Runs one search with noise off, and reports its visit counts and the move it chose.
#[pyfunction]
fn search(
    columns: Vec<u8>,
    exploration: f64,
    simulations: u32,
    evaluator: &Bound<'_, PyAny>,
) -> PyResult<(Vec<u32>, u8)> {
    let mut board = board_from(&columns)?;
    let mut tree = Search::new(simulations, exploration, 1.0, 0.25, false);
    let mut rng = PyRandom::from_text("noise is off, so this is never drawn from");

    tree.begin();
    while *tree.state() == SearchState::NeedsEvaluation {
        let answer = evaluator
            .call1((board.discs(board.turn()), board.discs(!board.turn())))?;
        let (priors, value): (Vec<f64>, f64) = answer.extract()?;
        let priors: [f64; POLICY_SIZE] = priors
            .try_into()
            .map_err(|_| PyValueError::new_err("the evaluator must return one prior per action"))?;
        tree.submit(&mut board, &priors, value, &mut rng);
    }

    let result = tree.result();
    let mut visits = vec![0u32; POLICY_SIZE];
    for slot in 0..result.children {
        visits[result.columns[slot] as usize] = result.visits[slot];
    }
    Ok((visits, result.most_visited()))
}

/// What the encoder makes of a position: the planes as nested lists, and the legal actions.
#[pyfunction]
fn encode(columns: Vec<u8>) -> PyResult<Encoded> {
    let board = board_from(&columns)?;
    let mut cells = [0u8; PLANE_BYTES];
    c4_core::encode::planes(&board, &mut cells);
    let mut legal = vec![false; POLICY_SIZE];
    c4_core::encode::legal_mask(&board, &mut legal);

    let nested = cells
        .chunks(ROWS as usize * COLS as usize)
        .map(|plane| {
            plane
                .chunks(COLS as usize)
                .map(|row| row.iter().map(|&cell| cell as i64).collect())
                .collect()
        })
        .collect();
    Ok((nested, legal))
}

#[pymodule]
fn zero_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SelfPlay>()?;
    module.add_function(wrap_pyfunction!(search, module)?)?;
    module.add_function(wrap_pyfunction!(encode, module)?)?;
    module.add_function(wrap_pyfunction!(best_move, module)?)?;
    module.add_function(wrap_pyfunction!(root_scores, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate, module)?)?;
    // What the encoder here produces, so a caller can refuse rather than reshape blindly.
    module.add("PLANE_SHAPE", (2, ROWS, COLS))?;
    module.add("POLICY_SIZE", POLICY_SIZE)?;
    module.add("GAME", "Connect4")?;
    // The scoring convention `best_move`/`root_scores` use, so a build with a different one -
    // or without the alpha-beta at all - is refused rather than silently misread.
    module.add("MATE", MATE)?;
    Ok(())
}
