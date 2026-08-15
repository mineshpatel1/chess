//! Many ladder games advanced together: a network-guided MCTS challenger against a fixed-depth
//! alpha-beta opponent, batched the way `selfplay.rs` batches self-play.
//!
//! Asymmetric where `selfplay.rs` is symmetric. Only the challenger's moves need a network
//! evaluation, so only the challenger's turns ever leave a game waiting on one - the opponent's
//! moves are `search::best_move`, resolved synchronously with no round trip at all. Both pieces
//! are reused exactly as they already are and are already parity-tested against their Python
//! originals; nothing here is a new algorithm, only the scheduling that puts many of these games
//! in flight together.
//!
//! A port of the shape `ai/ladder.py::climb` plays one rung in, via `ai/match.py::play_match`:
//! every opening is played twice, the challenger moving first in one game and second in the
//! other, and the tally is kept from the challenger's point of view.

use crate::connect4::Connect4;
use crate::encode::{legal_mask, planes, PLANE_BYTES, POLICY_SIZE};
use crate::mcts::{Search, State};
use crate::rng::PyRandom;
use crate::search::best_move;

pub struct Config {
    pub depth: i32,
    pub simulations: u32,
    pub exploration: f64,
    pub in_flight: usize,
}

/// The challenger's score, wins/draws/losses, exactly what `ai.match.MatchResult` holds.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tally {
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
}

/// One finished game: every move played from its opening, and how it ended.
pub struct Record {
    pub moves: Vec<u8>,
    pub winner: Option<bool>,
    pub challenger_is_first: bool,
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
    moves: Vec<u8>,
    challenger_is_first: bool,
    done: bool,
}

impl Game {
    /// `opening` is the position to start from; `challenger_is_first` is which seat the
    /// challenger plays, exactly `ai/match.py::play_match`'s `challenger_is_first in (True,
    /// False)` - both are always played, never drawn.
    fn start(index: usize, opening: &Connect4, challenger_is_first: bool, config: &Config) -> Self {
        let rng = PyRandom::from_text(&format!("ladder:{index}"));
        let mut board = opening.clone();
        let mut moves = Vec::new();
        let mut done = board.is_game_over();

        // Turns always alternate, so the opponent can be at most one move ahead of the challenger
        // at the start.
        if !done && board.turn() != challenger_is_first {
            let column = best_move(&mut board, config.depth).expect("a legal move exists");
            board.make_move(column);
            moves.push(column);
            done = board.is_game_over();
        }

        // Noise off, no Dirichlet: `model_player`'s ladder search reports its opinion, it does
        // not explore with it. `rng` is carried only because `Search::submit` asks for one; with
        // noise off it is never actually drawn from.
        let mut search = Search::new(config.simulations, config.exploration, 1.0, 0.25, false);
        if !done {
            search.begin();
        }

        Game { index, board, search, rng, moves, challenger_is_first, done }
    }

    /// Answers the position this game is waiting on, playing both the challenger's move and the
    /// opponent's reply if that finished the challenger's search.
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

        // Most-visited, not sampled: `MCTS.search(noise=False)` always reports its opinion
        // rather than exploring with it.
        let result = self.search.result();
        let column = result.most_visited();
        self.board.make_move(column);
        self.moves.push(column);
        if self.board.is_game_over() {
            self.done = true;
            return;
        }

        let opponent = best_move(&mut self.board, config.depth).expect("a legal move exists");
        self.board.make_move(opponent);
        self.moves.push(opponent);
        if self.board.is_game_over() {
            self.done = true;
            return;
        }

        self.search.begin();
    }

    fn finish(self) -> Record {
        let winner = self.board.result().expect("the game is not over").winner;
        Record {
            moves: self.moves,
            winner,
            challenger_is_first: self.challenger_is_first,
        }
    }
}

/// One rung of the ladder, played to a finish: every opening played twice, the challenger's own
/// turns batched across every game currently in flight.
pub struct LadderMatch {
    config: Config,
    queue: Vec<(Connect4, bool)>,
    cursor: usize,
    active: Vec<Game>,
    records: Vec<Option<Record>>,
    planes: Vec<u8>,
    legal: Vec<bool>,
}

impl LadderMatch {
    /// `openings` is the same list `ai/ladder.py::climb` reuses for every rung - its already
    /// selected, already paired `chosen` - played once with the challenger first and once with
    /// the challenger second, in that order.
    pub fn new(openings: Vec<Connect4>, config: Config) -> Self {
        let mut queue = Vec::with_capacity(openings.len() * 2);
        for opening in openings {
            queue.push((opening.clone(), true));
            queue.push((opening, false));
        }

        let in_flight = config.in_flight.clamp(1, queue.len().max(1));
        LadderMatch {
            active: Vec::with_capacity(in_flight),
            records: (0..queue.len()).map(|_| None).collect(),
            planes: Vec::with_capacity(in_flight * PLANE_BYTES),
            legal: Vec::with_capacity(in_flight * POLICY_SIZE),
            cursor: 0,
            queue,
            config: Config { in_flight, ..config },
        }
    }

    pub fn completed(&self) -> usize {
        self.records.iter().filter(|record| record.is_some()).count()
    }

    /// Starts games until the flight is full, then reports what they are all waiting on.
    ///
    /// `None` means every game has finished. The caller must evaluate the whole batch and hand it
    /// back through `submit` before asking again.
    pub fn pending(&mut self) -> Option<Batch<'_>> {
        while self.cursor < self.queue.len() && self.active.len() < self.config.in_flight {
            let (opening, challenger_is_first) = &self.queue[self.cursor];
            let game = Game::start(self.cursor, opening, *challenger_is_first, &self.config);
            self.cursor += 1;
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

    /// Answers the batch `pending` reported, in the same order. Single-threaded, for the same
    /// reason `SelfPlay::submit` is: the batch is where the parallelism pays, not the bookkeeping
    /// around it.
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
    }

    /// Every game's record, in the order `new`'s `openings` were queued.
    pub fn records(&self) -> impl Iterator<Item = &Record> {
        self.records.iter().map(|record| {
            record
                .as_ref()
                .expect("a game has not finished; drive `pending` to `None` first")
        })
    }

    /// The rung's score, from the challenger's point of view - exactly `ai.match.play_match`'s
    /// tally: a draw is half a point, a win the challenger's colour matches is a whole one.
    pub fn tally(&self) -> Tally {
        let mut tally = Tally::default();
        for record in self.records() {
            match record.winner {
                None => tally.draws += 1,
                Some(winner) if winner == record.challenger_is_first => tally.wins += 1,
                Some(_) => tally.losses += 1,
            }
        }
        tally
    }
}
