//! Many self-play games advanced together, so the positions they wait on can be evaluated at once.
//!
//! A port of `ai/zero/selfplay.py`. The shape is the one that module already argues for: one
//! tree's simulations cannot be batched, because each goes where the previous ones' statistics
//! send it, so the batching happens *between* games instead and every game gets the tree it would
//! have had alone. The only change here is the scale it can be run at - a Python batch is capped
//! by how many trees Python can afford to walk, and this one is capped by nothing much, which is
//! what lets a whole generation be in flight and a GPU be fed a batch it can use.
//!
//! Each game keeps its own random stream, seeded from the run's seed and the game's index, so how
//! many are in flight cannot change what is played.

use crate::connect4::Connect4;
use crate::encode::{legal_mask, planes, PLANE_BYTES, POLICY_SIZE};
use crate::mcts::{Search, State};
use crate::rng::PyRandom;

pub struct Config {
    pub games: usize,
    pub in_flight: usize,
    pub simulations: u32,
    pub exploration: f64,
    pub dirichlet_alpha: f64,
    pub dirichlet_epsilon: f64,
    pub temperature: f64,
    pub final_temperature: f64,
    pub temperature_moves: usize,
    pub opening_plies: u32,
    pub seed: String,
}

/// One finished game's training examples, and how it ended.
pub struct Record {
    pub planes: Vec<u8>,
    pub policies: Vec<f64>,
    pub values: Vec<f64>,
    pub positions: usize,
    pub winner: Option<bool>,
}

/// The positions every game in flight is waiting on, in slot order.
pub struct Batch<'a> {
    pub planes: &'a [u8],
    pub legal: &'a [bool],
    pub positions: usize,
}

struct Game {
    index: usize,
    board: Connect4,
    search: Search,
    rng: PyRandom,
    planes: Vec<u8>,
    policies: Vec<f64>,
    movers: Vec<bool>,
    done: bool,
}

impl Game {
    fn start(index: usize, config: &Config) -> Self {
        let mut rng = PyRandom::from_text(&format!("{}:{}", config.seed, index));
        let board = opening(config.opening_plies, &mut rng);
        let done = board.is_game_over();

        let mut game = Game {
            index,
            board,
            search: Search::new(
                config.simulations,
                config.exploration,
                config.dirichlet_alpha,
                config.dirichlet_epsilon,
                true,
            ),
            rng,
            planes: Vec::new(),
            policies: Vec::new(),
            movers: Vec::new(),
            done,
        };
        if !game.done {
            game.search.begin();
        }
        game
    }

    /// Answers the position this game is waiting on, and plays a move if that finished a search.
    fn advance(&mut self, priors: &[f32], value: f32, config: &Config) {
        let mut widened = [0.0f64; POLICY_SIZE];
        for (slot, prior) in priors.iter().enumerate() {
            widened[slot] = *prior as f64;
        }
        self.search
            .submit(&mut self.board, &widened, value as f64, &mut self.rng);

        if *self.search.state() != State::Finished {
            return;
        }

        let result = self.search.result();

        // Recorded before the move: the example belongs to the position it was searched from.
        let start = self.planes.len();
        self.planes.resize(start + PLANE_BYTES, 0);
        planes(&self.board, &mut self.planes[start..]);
        self.policies.extend_from_slice(&result.policy());
        self.movers.push(self.board.turn());

        let temperature = if self.movers.len() <= config.temperature_moves {
            config.temperature
        } else {
            config.final_temperature
        };
        self.board.make_move(result.sample(temperature, &mut self.rng));

        if self.board.is_game_over() {
            self.done = true;
        } else {
            self.search.begin();
        }
    }

    fn finish(self) -> Record {
        let winner = self.board.result().expect("the game is not over").winner;
        let values = self
            .movers
            .iter()
            .map(|&mover| value_to(mover, winner))
            .collect();

        Record {
            planes: self.planes,
            policies: self.policies,
            values,
            positions: self.movers.len(),
            winner,
        }
    }
}

pub struct SelfPlay {
    config: Config,
    active: Vec<Game>,
    started: usize,
    completed: usize,
    records: Vec<Option<Record>>,
    planes: Vec<u8>,
    legal: Vec<bool>,
}

impl SelfPlay {
    pub fn new(config: Config) -> Self {
        let in_flight = config.in_flight.clamp(1, config.games.max(1));
        SelfPlay {
            active: Vec::with_capacity(in_flight),
            started: 0,
            completed: 0,
            records: (0..config.games).map(|_| None).collect(),
            planes: Vec::with_capacity(in_flight * PLANE_BYTES),
            legal: Vec::with_capacity(in_flight * POLICY_SIZE),
            config: Config { in_flight, ..config },
        }
    }

    pub fn completed(&self) -> usize {
        self.completed
    }

    /// Starts games until the flight is full, then reports what they are all waiting on.
    ///
    /// `None` means every game has finished. The caller must evaluate the whole batch and hand it
    /// back through `submit` before asking again.
    pub fn pending(&mut self) -> Option<Batch<'_>> {
        while self.started < self.config.games && self.active.len() < self.config.in_flight {
            let game = Game::start(self.started, &self.config);
            self.started += 1;
            if game.done {
                self.retire(game);
            } else {
                self.active.push(game);
            }
        }

        if self.active.is_empty() {
            return None;
        }

        self.planes.clear();
        self.planes.resize(self.active.len() * PLANE_BYTES, 0);
        self.legal.clear();
        self.legal.resize(self.active.len() * POLICY_SIZE, false);

        for (slot, game) in self.active.iter().enumerate() {
            planes(
                &game.board,
                &mut self.planes[slot * PLANE_BYTES..(slot + 1) * PLANE_BYTES],
            );
            legal_mask(
                &game.board,
                &mut self.legal[slot * POLICY_SIZE..(slot + 1) * POLICY_SIZE],
            );
        }

        Some(Batch {
            planes: &self.planes,
            legal: &self.legal,
            positions: self.active.len(),
        })
    }

    /// Answers the batch `pending` reported, in the same order.
    ///
    /// Single-threaded on purpose. The games are independent and could be advanced across
    /// threads, but one game's share of a batch is a few hundred nanoseconds against tens of
    /// thousands of batches, so the synchronisation costs several times what the work does. The
    /// parallelism that pays here is the batch itself, and that happens on the GPU.
    pub fn submit(&mut self, priors: &[f32], values: &[f32]) {
        assert_eq!(priors.len(), self.active.len() * POLICY_SIZE);
        assert_eq!(values.len(), self.active.len());

        let config = &self.config;
        self.active
            .iter_mut()
            .zip(priors.chunks(POLICY_SIZE))
            .zip(values.iter())
            .for_each(|((game, priors), &value)| game.advance(priors, value, config));

        let mut slot = 0;
        while slot < self.active.len() {
            if self.active[slot].done {
                let game = self.active.swap_remove(slot);
                self.retire(game);
            } else {
                slot += 1;
            }
        }
    }

    fn retire(&mut self, game: Game) {
        let index = game.index;
        self.records[index] = Some(game.finish());
        self.completed += 1;
    }

    /// Every game's examples, in game order rather than the order they happened to finish in.
    pub fn records(&self) -> impl Iterator<Item = &Record> {
        self.records.iter().map(|record| {
            record
                .as_ref()
                .expect("a game has not finished; drive `pending` to `None` first")
        })
    }

    pub fn positions(&self) -> usize {
        self.records
            .iter()
            .filter_map(|record| record.as_ref())
            .map(|record| record.positions)
            .sum()
    }
}

/// A position a random number of random plies in, and still running. Zero plies - the measured
/// default - is the empty board and draws nothing from the stream.
fn opening(plies: u32, rng: &mut PyRandom) -> Connect4 {
    if plies == 0 {
        return Connect4::new();
    }

    let mut moves = [0u8; POLICY_SIZE];
    for _ in 0..50 {
        let mut board = Connect4::new();
        for _ in 0..rng.below(plies + 1) {
            if board.is_game_over() {
                break;
            }
            let count = board.legal_moves(&mut moves);
            board.make_move(moves[rng.below(count as u32) as usize]);
        }

        if !board.is_game_over() {
            return board;
        }
    }
    Connect4::new()
}

/// What a finished game was worth to the player who was about to move.
fn value_to(mover: bool, winner: Option<bool>) -> f64 {
    match winner {
        None => 0.0,
        Some(winner) => {
            if winner == mover {
                1.0
            } else {
                -1.0
            }
        }
    }
}
