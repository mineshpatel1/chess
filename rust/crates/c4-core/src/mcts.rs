//! A PUCT search over Connect 4, suspended at every position it needs evaluated.
//!
//! A port of `ai/zero/mcts.py`, and a deliberately literal one: the arithmetic is f64 in the same
//! order, children hang off a parent in the order moves are generated, and ties break by that
//! order. The tree is still made of paths rather than positions, still discarded between plies,
//! and still takes one leaf at a time - none of which is an oversight, and all of which is what
//! makes the two implementations comparable.
//!
//! Where the Python is a generator, this is a state machine over the same three moments: hand it
//! a position, take the leaf it wants evaluated, hand back the answer.

use crate::connect4::Connect4;
use crate::constants::COLS;
use crate::encode::POLICY_SIZE;
use crate::rng::PyRandom;

/// One edge of the tree. Children are contiguous, so a node needs an index and a count rather
/// than a map from move to child.
#[derive(Clone, Copy)]
struct Node {
    prior: f64,
    value_sum: f64,
    visits: u32,
    first_child: u32,
    children: u8,
    column: u8,
}

impl Node {
    fn new(prior: f64, column: u8) -> Self {
        Node {
            prior,
            value_sum: 0.0,
            visits: 0,
            first_child: 0,
            children: 0,
            column,
        }
    }
}

/// What a search produced, with the root's children in the order they were generated.
pub struct SearchResult {
    pub columns: [u8; POLICY_SIZE],
    pub visits: [u32; POLICY_SIZE],
    pub children: usize,
}

impl SearchResult {
    /// Visit counts as a distribution over the whole action space: the training target.
    pub fn policy(&self) -> [f64; POLICY_SIZE] {
        let mut policy = [0.0; POLICY_SIZE];
        let total: u32 = self.visits[..self.children].iter().sum();
        if total == 0 {
            return policy;
        }

        for slot in 0..self.children {
            policy[self.columns[slot] as usize] = self.visits[slot] as f64 / total as f64;
        }
        policy
    }

    /// The move tried most often, ties broken by generation order.
    pub fn most_visited(&self) -> u8 {
        let mut best = (0u8, -1i64);
        for slot in 0..self.children {
            if self.visits[slot] as i64 > best.1 {
                best = (self.columns[slot], self.visits[slot] as i64);
            }
        }
        best.0
    }

    /// A move drawn from the visit counts, for exploration early in a self-play game.
    pub fn sample(&self, temperature: f64, rng: &mut PyRandom) -> u8 {
        if temperature <= 0.0 {
            return self.most_visited();
        }

        let weights: Vec<f64> = self.visits[..self.children]
            .iter()
            .map(|&count| (count as f64).powf(1.0 / temperature))
            .collect();
        let total: f64 = weights.iter().sum();

        if total <= 0.0 {
            return self.columns[rng.below(self.children as u32) as usize];
        }
        self.columns[rng.weighted_index(&weights)]
    }
}

/// Whether the search wants another position evaluated, or has run its simulations out.
#[derive(PartialEq, Eq, Debug)]
pub enum State {
    NeedsEvaluation,
    Finished,
}

pub struct Search {
    arena: Vec<Node>,
    path: Vec<u32>,
    simulations: u32,
    exploration: f64,
    dirichlet_alpha: f64,
    dirichlet_epsilon: f64,
    noise: bool,
    completed: u32,
    root_expanded: bool,
    state: State,
}

impl Search {
    pub fn new(
        simulations: u32,
        exploration: f64,
        dirichlet_alpha: f64,
        dirichlet_epsilon: f64,
        noise: bool,
    ) -> Self {
        Search {
            arena: Vec::with_capacity(simulations as usize * 2 + COLS as usize),
            path: Vec::with_capacity(COLS as usize * crate::constants::ROWS as usize + 1),
            simulations,
            exploration,
            dirichlet_alpha,
            dirichlet_epsilon,
            noise,
            completed: 0,
            root_expanded: false,
            state: State::NeedsEvaluation,
        }
    }

    /// Points the search at a position. The position itself is what it wants evaluated first.
    pub fn begin(&mut self) {
        self.arena.clear();
        self.arena.push(Node::new(1.0, u8::MAX));
        self.path.clear();
        self.path.push(0);
        self.completed = 0;
        self.root_expanded = false;
        self.state = State::NeedsEvaluation;
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    /// Answers the position the search is sitting on, then runs until it needs another or is done.
    ///
    /// `board` is the live position: on entry it is the leaf being answered, and on return it is
    /// either the next leaf or back at the root. Priors are indexed by action, not by child.
    pub fn submit(
        &mut self,
        board: &mut Connect4,
        priors: &[f64; POLICY_SIZE],
        value: f64,
        rng: &mut PyRandom,
    ) {
        debug_assert_eq!(self.state, State::NeedsEvaluation);
        let leaf = *self.path.last().expect("no leaf to answer");
        self.expand(board, leaf, priors);

        if self.root_expanded {
            self.back_up(value);
            self.unwind(board);
            self.completed += 1;
        } else {
            // Expanding counts as the root's first visit, so the first selection sees sqrt(1).
            self.arena[0].value_sum = value;
            self.arena[0].visits = 1;
            self.root_expanded = true;
            if self.noise {
                self.add_noise(rng);
            }
        }

        self.descend(board);
    }

    pub fn result(&self) -> SearchResult {
        let root = self.arena[0];
        let mut result = SearchResult {
            columns: [0; POLICY_SIZE],
            visits: [0; POLICY_SIZE],
            children: root.children as usize,
        };

        for slot in 0..root.children as usize {
            let child = self.arena[root.first_child as usize + slot];
            result.columns[slot] = child.column;
            result.visits[slot] = child.visits;
        }
        result
    }

    // ---- the tree ------------------------------------------------------------------------

    /// Walks to the next leaf that needs evaluating, playing out terminal ones on the way.
    fn descend(&mut self, board: &mut Connect4) {
        while self.completed < self.simulations {
            self.path.truncate(1);

            loop {
                let node = *self.path.last().expect("the root is always on the path");

                if board.is_game_over() {
                    let value = board.terminal_value();
                    self.back_up(value);
                    self.unwind(board);
                    self.completed += 1;
                    break;
                }

                if self.arena[node as usize].children == 0 {
                    self.state = State::NeedsEvaluation;
                    return;
                }

                let child = self.select(node);
                board.make_move(self.arena[child as usize].column);
                self.path.push(child);
            }
        }

        self.state = State::Finished;
    }

    /// The child maximising `Q + c * P * sqrt(N_parent) / (1 + N_child)`.
    ///
    /// `-value` because a child holds its value from *its* mover's point of view, and the player
    /// choosing here is the other one.
    fn select(&self, node: u32) -> u32 {
        let parent = self.arena[node as usize];
        let parent_visits = if parent.visits != 0 {
            (parent.visits as f64).sqrt()
        } else {
            0.0
        };

        let mut best_score = f64::NEG_INFINITY;
        let mut best = parent.first_child;

        for slot in 0..parent.children as u32 {
            let child = self.arena[(parent.first_child + slot) as usize];
            let exploit = if child.visits != 0 {
                -(child.value_sum / child.visits as f64)
            } else {
                0.0
            };
            let explore =
                self.exploration * child.prior * parent_visits / (1.0 + child.visits as f64);
            let score = exploit + explore;

            if score > best_score {
                best_score = score;
                best = parent.first_child + slot;
            }
        }
        best
    }

    /// Hangs a node's legal moves off it, with the priors renormalised over those moves only:
    /// the evaluator may spend mass on moves that do not exist here, and the search must not.
    fn expand(&mut self, board: &Connect4, node: u32, priors: &[f64; POLICY_SIZE]) {
        let mut moves = [0u8; COLS as usize];
        let count = board.legal_moves(&mut moves);

        let mut weights = [0.0f64; COLS as usize];
        let mut total = 0.0;
        for slot in 0..count {
            let prior = priors[moves[slot] as usize];
            weights[slot] = if 0.0 > prior { 0.0 } else { prior }; // `max`, and its NaN handling
            total += weights[slot];
        }

        if total <= 0.0 {
            // An evaluator with no opinion, or one that masked everything away.
            weights[..count].fill(1.0);
            total = count as f64;
        }

        let first_child = self.arena.len() as u32;
        for slot in 0..count {
            self.arena
                .push(Node::new(weights[slot] / total, moves[slot]));
        }
        self.arena[node as usize].first_child = first_child;
        self.arena[node as usize].children = count as u8;
    }

    /// Dirichlet noise over the root's children, so self-play does not play one game repeatedly.
    fn add_noise(&mut self, rng: &mut PyRandom) {
        let root = self.arena[0];
        if root.children == 0 {
            return;
        }

        let noise: Vec<f64> = (0..root.children)
            .map(|_| rng.gammavariate(self.dirichlet_alpha, 1.0))
            .collect();
        let mut total: f64 = noise.iter().sum();
        if total == 0.0 {
            total = 1.0;
        }

        let epsilon = self.dirichlet_epsilon;
        for (slot, sample) in noise.iter().enumerate() {
            let child = &mut self.arena[root.first_child as usize + slot];
            child.prior = (1.0 - epsilon) * child.prior + epsilon * sample / total;
        }
    }

    /// One value, negated at every step back up, because each node holds its own mover's view.
    fn back_up(&mut self, value: f64) {
        let mut value = value;
        for &node in self.path.iter().rev() {
            let node = &mut self.arena[node as usize];
            node.visits += 1;
            node.value_sum += value;
            value = -value;
        }
    }

    fn unwind(&self, board: &mut Connect4) {
        for _ in 1..self.path.len() {
            board.unmake_move();
        }
    }
}
