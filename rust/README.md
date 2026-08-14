# The Rust Connect 4 engine

Connect 4, fast enough to feed a GPU and to grade what it learns. Three ports, sharing one board:
self-play (below), [the ladder's alpha-beta opponents](#the-alpha-beta), and
[the ladder's network challenger](#the-ladder) that plays the two of them against each other.
Optional, in exactly the way PyTorch is optional: the engine, the tests and a training run all
work without it, and `zero.py train --engine` is what asks for it.

It exists because a 2,000-game generation at 600 simulations took three and a half hours, and the
network was the reason only in the sense that it was being fed one position per game per pass.
The Python driver evaluates a batch capped by how many trees Python can afford to walk - 32 by
default, 128 before the tree work outgrows the forward pass - while the card will take 4,096.
Moving the board, the tree and the driver to Rust lifts that cap to *every game in the
generation*, and the tree work stops being a cost at all.

| | Python | Rust |
|---|---|---|
| Board, encoder, PUCT tree | `games/connect4/`, `ai/zero/mcts.py` | `crates/c4-core/` |
| Games in flight | `ai/zero/selfplay.py`, batch ≤ `--games-in-flight` | `crates/c4-core/src/selfplay.rs`, batch = `--games` |
| The network | PyTorch | PyTorch, unchanged |

**The network never crosses the boundary.** Rust hands out planes and takes back priors and
values, so there is no second copy of the model, nothing to export between generations, and no
numerical parity to argue about. `crates/zero-rs/` is the twenty lines of PyO3 that make that a
Python object.

## Building

Needs a Rust toolchain (<https://rustup.rs>) and a C linker, plus `maturin` in whichever
environment has PyTorch:

```bash
pip install maturin
maturin develop --release -m rust/crates/zero-rs/Cargo.toml
python3 -c 'import zero_rs; print(zero_rs.GAME, zero_rs.PLANE_SHAPE)'
```

```bash
python3 zero.py --game connect4 train --engine rust --device cuda ...
```

`--engine auto` is the default and takes this engine when it is built; `--engine rust` insists on
it, because a run started for the speed should not quietly take four hours instead. `--engine
python` is always available and is how the two are compared.

`ai/zero/fast.py` is the Python side of it: `available()` says whether the build is there and
speaks for the game and the encoder in front of it, `why_unavailable()` gives the log line when it
is not, and `play_games(net, count, simulations, ...)` plays a generation and hands back the three
stacked arrays `ai/zero/replay.py` already stores.

**`--games-in-flight` means something different on each engine.** For the Python driver it is a
cap on how many trees it can afford to walk, and it defaults to 32. This one has no such cap, so
it defaults to the whole generation - which is the point of it, since the batch is what the card
is for.

## The two engines play the same games

Not "equivalent games" - the same ones, example for example. That is the whole design constraint,
because a speed change nobody can check is a change of player nobody noticed.

The search is a literal port: f64 in the same order, children in the order moves are generated,
ties broken by that order, no tree reuse, no virtual loss, no in-tree win shortcut. The random
numbers are CPython's - Mersenne Twister, `random()`'s two 32-bit draws, `gammavariate` and
`choices` - so the Dirichlet noise at each root and the move sampled from the visit counts match
too, down to `random.Random('1:37')` seeding through SHA-512.

```bash
cargo test --manifest-path rust/Cargo.toml    # the board, the RNG, the tree, the driver
python3 -m unittest tests.zero.test_fast -v   # both engines, side by side
```

`tests/zero/test_fast.py` runs the comparison live: the same evaluator into both searches must
give identical visit counts, and a generation played by each with the same network and seed must
give identical planes, policy targets and value targets. `crates/c4-core/tests/search.rs` pins
the same answers so `cargo test` can check them without an interpreter; regenerate them with
`python3 -m tests.zero.test_fast --write-fixture` if the search ever legitimately changes.

## The alpha-beta

The other cost once self-play stopped being one: `train._climb` plays the network up a ladder of
`minimax:N` opponents every few generations, and `minimax:7`/`:8` at 100 games each is most of a
generation once self-play is minutes rather than hours. `ai.search.alpha_beta` is a plain
fixed-depth negamax with no transposition table and no move ordering beyond the board's own
centre-first `legal_moves` - which makes it a small, strictly provable port: `c4-core::search` and
`c4-core::evaluation` are the same ~90 lines with nothing added, not a table, not a killer move,
not an ordering the Python search does not have. Adding any of those would be real speed and would
also change which move comes back, and that is a different, separately measured piece of work.

```bash
python3 bench.py search --depths 6 7 8    # what a ladder rung costs, per engine
python3 zero.py --game connect4 ladder --engine rust --player model:PATH --rungs minimax:7 minimax:8
```

`ai/native.py` is the Python side, in the same three-part shape as `ai/zero/fast.py`:
`available()`, `why_unavailable()`, and `alpha_beta(state, depth, evaluate=None)` - which refuses
a custom `evaluate` rather than silently ignoring it, since the tests that pass one need the
Python search's ordering, not a faster copy of it. `ai.players.player(spec, engine=...)` is where
every `minimax:` clause resolves, so `zero.py train --engine rust` means the Rust search for both
self-play and the ladder, and `play.py`'s computer opponent gets it for free.

`tests/connect4/test_native.py` compares every root move's score at every depth over the same
corpus `tests/connect4/test_search_equivalence.py` already uses, and pins a fixture so
`cargo test` can check the same answers - scores and the number of leaves reached - without an
interpreter; regenerate it with `python3 -m tests.connect4.test_native --write-fixture`.

## The ladder

The alpha-beta made the opponent free; this is the half of the ladder that was left; the
challenger's own MCTS. `ai.zero.train._climb` played it one game at a time in Python - a batch of
one forward pass per network call, which is why it ran on the CPU rather than the GPU a training
run otherwise uses: a batch of one costs more on a card than off it. `crates/c4-core/src/ladder.rs`
is the same fix self-play already got, aimed at the ladder instead: every game a rung needs *in
flight* together, so the network's forward passes batch across games and a card is worth using
again. It is asymmetric where `selfplay.rs` is symmetric - only the challenger's moves need a
network evaluation, so only the challenger's turns ever leave a game waiting on one; the opponent's
`minimax:N` replies resolve with `search::best_move`, synchronously, no round trip back to Python
at all. Nothing here is a new algorithm: `mcts::Search` and `search::best_move` are reused exactly
as they already are and are already parity-tested against their Python originals by the two ports
above, so this is a scheduler in the shape `selfplay.rs` already is, not a third search to prove.

```bash
python3 bench.py ladder --simulations 100 300 600   # what a ladder rung's challenger costs
python3 zero.py --game connect4 ladder --engine rust --player model:PATH+mcts:600 --rungs minimax:7 minimax:8
```

| games vs `minimax:7`, 100 each | python | rust | speedup |
|---|---|---|---|
| 100 simulations | 196.2s | 6.5s | 30.3× |
| 300 simulations | 426.9s | 13.2s | 32.4× |

600 simulations - the reference paper's own challenger, and the number that made this worth
building - was not run to completion in Python on this machine: at roughly twice 300's cost, it is
the better part of fifteen minutes for one row of one table, and the trend across 100 and 300
already says what it would show. It is the Rust column that answers the question that mattered:
seconds, not minutes, whatever the simulation count.

`ai/zero/ladder_fast.py` is the Python side, in the same `pending`/`submit` shape `ai/zero/fast.py`
already drives: `available()`, `why_unavailable()`, and `climb(net, encoder, depth, chosen,
simulations, ...)` plays one rung and returns an `ai.match.MatchResult`. `ai.ladder.climb` is where
it is reached from - inside the existing per-rung loop, `_rung_result` takes this path when the
rung is `minimax:N`, the challenger is a network carrying enough to search with (`.net`/`.encoder`
from a network already in memory, or `.checkpoint` from one loaded off disk - both are tried, so a
training run's ladder check and a standalone `zero.py ladder` take the same fast path), and
`engine` allows it - falling back to `play_match`, one game at a time, for anything that does not
fit that shape (`'random'`, a raw-policy challenger, `engine='python'`). `engine='rust'` insists on
it here exactly as it already does for the alpha-beta, rather than quietly taking the slow path.

```bash
cargo test --manifest-path rust/Cargo.toml --test ladder     # the driver against itself
python3 -m unittest tests.zero.test_ladder_fast -v            # the pinned fixture, and _climb live
```

Two parity claims, proven two different ways. The *game* - challenger MCTS plus alpha-beta
opponent plus the pairing and tally `ai.match.play_match` already defines - is checked with the
same hashed evaluator `tests/zero/test_fast.py` pins its own fixture with, so no network or torch
is needed: a Python reference plays it move for move, `rust/crates/c4-core/tests/fixtures/
ladder_fixture.rs` pins the answers, and `tests/ladder.rs` replays them against an unbatched
reference in Rust that the same test file separately proves plays identically to the batched
driver - regenerate with `python3 -m tests.zero.test_ladder_fast --write-fixture`. The
*integration* - `ai.zero.train._climb` itself, with a real network - is `test_ladder_fast.py`'s
live comparison, `engine='python'` against `engine='rust'`, both pinned to the CPU: a GPU batched
forward pass is not guaranteed bit-identical to a CPU single-position one, so a live GPU run is a
strength match against the Python path rather than a move-for-move-identical one.

## Layout

```
crates/c4-core/          no Python, no torch, no unsafe
  src/constants.rs       the board's shape, and every mask derived from it
  src/bitboard.rs        the carry that drops a disc, the shifts that find four in a row
  src/connect4.rs        two integers, make/unmake, win detection
  src/encode.rs          two planes of 6x7, seven shared actions
  src/evaluation.rs      open threes, weighted by direction and by whether they can be played
  src/ladder.rs          the challenger's MCTS against the alpha-beta, games in flight
  src/mcts.rs            the PUCT tree, as a state machine over the same three moments
  src/rng.rs             CPython's random.Random
  src/search.rs          fixed-depth alpha-beta, the ladder's minimax:N opponents
  src/selfplay.rs        games in flight, batch out and batch in
crates/zero-rs/          the PyO3 module: SelfPlay, LadderMatch, the alpha-beta hooks, the parity
                         hooks
```

`c4-core` depends on `sha2` alone, and only to seed the Mersenne Twister the way
`random.Random(text)` does.

Advancing the games is single-threaded. Rayon was tried and cost three times what it saved: one
game's share of a batch is a few hundred nanoseconds against twenty thousand batches a
generation, so synchronisation dominates. The parallelism that pays here is the batch, and that
happens on the GPU.
