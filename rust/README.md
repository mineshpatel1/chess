# The Rust self-play engine

Connect 4 self-play, fast enough to feed a GPU. Optional, in exactly the way PyTorch is optional:
the engine, the tests and a training run all work without it, and `zero.py train --engine` is
what asks for it.

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

## Layout

```
crates/c4-core/          no Python, no torch, no unsafe
  src/constants.rs       the board's shape, and every mask derived from it
  src/bitboard.rs        the carry that drops a disc, the shifts that find four in a row
  src/connect4.rs        two integers, make/unmake, win detection
  src/encode.rs          two planes of 6x7, seven shared actions
  src/mcts.rs            the PUCT tree, as a state machine over the same three moments
  src/rng.rs             CPython's random.Random
  src/selfplay.rs        games in flight, batch out and batch in
crates/zero-rs/          the PyO3 module: SelfPlay, and two hooks for the parity test
```

`c4-core` depends on `sha2` alone, and only to seed the Mersenne Twister the way
`random.Random(text)` does.

Advancing the games is single-threaded. Rayon was tried and cost three times what it saved: one
game's share of a batch is a few hundred nanoseconds against twenty thousand batches a
generation, so synchronisation dominates. The parallelism that pays here is the batch, and that
happens on the GPU.
