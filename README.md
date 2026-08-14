# AlphaZero and Board Games

Game engines written from scratch in pure Python — bitboard move generation, alpha-beta search,
and a UCI interface for the chess. No third-party runtime dependencies.

Three games share one search, one move enumerator and one conformance suite, and `ai/` contains no
chess at all. Each has its own page:

| Game | What it adds |
|---|---|
| [**Chess**](games/chess/README.md) | The reason the project exists. Bitboards, a UCI engine and GUI integration, perft against the reference counts, and matches against third-party engines |
| [**Connect 4**](games/connect4/README.md) | A sentinel-padded bitboard, an exact solver, 33,300 committed solved positions, and a network that beats a seven-ply search |
| [**Tic-tac-toe**](games/tictactoe/README.md) | **Solved**, so every player can be graded against the truth in all 5,478 positions — which is what makes it the game the learned player is developed on |

## Quick start

Requires Python 3.7+. Nothing to install.

```bash
python3 -m unittest tests.test_all -v                       # the test suite
python3 play.py                                             # play any game in the terminal
python3 -m games.chess.uci                                  # the UCI engine on stdin
python3 zero.py benchmark --player minimax:9                # grade a player against perfect play
python3 zero.py --game connect4 ladder --player minimax:4   # play it against a set of opponents
```

The learned player is the one exception to "nothing to install", and it is opt-in:

```bash
pip install -r requirements-zero.txt      # PyTorch, and only for ai/zero/
python3 zero.py train                     # train a tic-tac-toe network from scratch
```

## How it works

The search asks a game for its legal moves, plays them, unplays them, and asks how a finished
position finished. Everything else — chess's bitboards, Connect 4's carry, tic-tac-toe's eight
lines — lives behind that contract in `games/`.

| Module | What is in it |
|---|---|
| `games/base.py` | The `GameState` contract, and the optional `Encoder` that says what a network sees |
| `games/chess/`, `games/connect4/`, `games/tictactoe/` | The three games, each with its own README |
| `ai/search.py` | Negamax with alpha-beta pruning, and a random mover |
| `ai/perft.py`, `ai/simulate.py` | Move enumeration, and playing two move-choosers off |
| `ai/match.py` | Doing that a few hundred times with paired openings, which is how an evaluation is judged |
| `ai/ladder.py` | Placing a player against a fixed sequence of opponents — can it be beaten |
| `ai/oracle.py` | Exact play, and grading any player against it over a set of positions |
| `ai/corpus.py`, `ai/generate.py` | Reading positions whose exact value was computed once, and choosing them |
| `ai/players.py` | Naming a player — `minimax:9`, `model:best.pt+mcts:200` — in one place |
| `ai/zero/` | The AlphaZero implementation: `mcts.py`, `net.py`, `selfplay.py`, `train.py`, `metrics.py` |
| `play.py`, `zero.py`, `plot.py`, `bench.py` | Terminal front end; training and grading; metrics charts; solver timings |
| `tests/conformance.py` | The tests every game must pass |

## Adding a game

A move can be whatever suits the game — the search only ever hands one back.

1. **Implement `GameState`** (`games/base.py`) in `games/<name>/`: `legal_moves`, `make_move`,
   `unmake_move`, `copy`, plus `outcome_without_moves`, which says what it means to have nothing to
   play. In chess that is checkmate or stalemate; in a game that fills up, it is a draw.
2. **Override `outcome`** if the game can be won while moves remain. It is checked at every node,
   so keep it cheap; Connect 4 tests only the board of the player who just moved, because only that
   player can have just won. It defaults to `None`, which is right for chess. `ai/perft.py` honours
   it, so a decided game counts as having no continuations.
3. **Write an evaluation** returning a score positive for the player to move — the search is
   negamax, so every score is read from the point of view of whoever is on move. Name it as the
   class's `DEFAULT_EVAL` and keep it well inside `ai.search.MATE`. Justify it with `ai/match.py`
   against a *fixed* opponent rather than by eye or against its own previous version; see
   [what the Connect 4 evaluation is worth](games/connect4/README.md#what-the-evaluation-is-worth)
   for how badly that distinction bites.
4. **Generate moves in a useful order.** `legal_moves` may return them in any order and the search
   takes them in that order, so ordering is the game's business and is most of what makes
   alpha-beta pay.
5. **Set `PARALLEL_ROOT`** if the game branches widely enough that splitting the root across a
   process pool beats starting one. Chess does; a game with seven or nine moves does not.
6. **Override `parse_move`** only to say *why* a move typed at `play.py` was refused; the default
   matches what was typed against `str(move)`.
7. **Set `SOLVED_DEPTH`** if the whole game fits inside a search, as tic-tac-toe's nine plies do —
   `play.py` uses it as its default depth, so a solved game arrives at the prompt playing perfectly
   instead of merely well. Leave it `None` otherwise.
8. **Register it** in `games/__init__.py` and **subclass `GameConformanceTests`**
   (`tests/conformance.py`), filling in its three hooks. That suite catches the mistakes that do not
   look like mistakes — an undo that restores almost everything, a copy that shares state with its
   original.

Optionally, **supply an `ENCODER`** — planes and a shared action space — if the game should be
learnable by `ai/zero/`. Connect 4 and tic-tac-toe have one; chess does not.

Nothing in `ai/` should need to change: adding Connect 4 changed one thing there, and adding
tic-tac-toe changed nothing.

## Learning to play

`ai/zero/` is an AlphaZero implementation: a network learns from nothing but games against itself.
No opening book, no evaluation function, no opponent — the only fact from outside the process is
who won. It needs PyTorch, and it is the only thing here that does; the learning tests skip
themselves when torch is absent, so `tests.test_all` passes on a clean checkout either way.

```bash
pip install -r requirements-zero.txt
python3 zero.py [--game connect4] {train,benchmark,ladder,match} ...
```

Trained checkpoints are committed, so it all works immediately after cloning:
`models/tictactoe-best.pt`, `models/connect4-latest.pt` and `models/connect4-g2000-latest.pt`, with
their metrics in `runs/`.

### Naming a player

`ai/players.py` turns a string into the `state -> move` callable every harness expects, so the
terminal, the benchmark, the match harness and the ladder all mean the same thing by an opponent.

| Spec | Player |
|---|---|
| `random` | a uniformly random legal move |
| `minimax` / `minimax:9` | alpha-beta at the game's default depth, or a given one |
| `model:PATH` | the network's policy, **with no search at all** |
| `model:PATH+mcts:200` | the same network, thinking 200 simulations ahead |
| `human` | read a move from the terminal |

The last two are the reason it exists: raw intuition and intuition-plus-search are one clause
apart, so comparing them is a change of argument rather than of code. A network that is only strong
at 500 simulations has learned to be a prior for a search doing the real work; one that is strong at
zero has learned the game.

### Training

`zero.py train` runs the generation loop: self-play, then gradient steps on a replay buffer, then
grading. The defaults are tuned for tic-tac-toe, which trains in about 45 minutes:

```bash
python3 zero.py train \
    --generations 400 \        # every value here is also the default
    --games 80 \               # self-play games per generation
    --simulations 50 \         # MCTS simulations per move
    --steps 60 \               # gradient steps per generation
    --seed 1 \
    --out models/tictactoe-best.pt
```

Training is deterministic given `--seed`. The rest of the configuration lives in `ai/zero/train.py`
and `ai/zero/selfplay.py`; what each value was measured at is here:

| Parameter | Default | Why |
|---|---|---|
| `SELF_PLAY_EXPLORATION` (c_puct) | **5.0** | An inverted U with a clear peak — 1.5/3.0/5.0/8.0 → 92.8/95.4/95.8/94.7% |
| `SIMULATIONS` | **50** | 50→200 buys 1.0 point for 3.1× the time |
| `TEMPERATURE_MOVES` | 30 (the whole game) | Cutting to 3 plies cost 10 points |
| `OPENING_PLIES` | **0** | Random starts were covering for c_puct being too low |
| `SYMMETRIES` | **False** | No benefit, and it is knowledge the network should not be given |
| `DIRICHLET_EPSILON` / `ALPHA` | 0.25 / 1.0 | Root noise, self-play only — never at evaluation |
| `BUFFER_SIZE` | 20,000 positions | Scale it with `--games` or a generation overflows it |
| `GAMES_IN_FLIGHT` | 32 | Throughput only — see [Batching](#batching) |
| Optimiser | Adam, lr 1e-3, decay 1e-4, batch 128 | |

Those exploration values were measured on tic-tac-toe; the
[sweeps are on its page](games/tictactoe/README.md#reproducing-the-committed-checkpoint). The
network architecture is per game and lives with the game — dense layers for
[tic-tac-toe](games/tictactoe/README.md#the-learned-player), a residual tower for
[Connect 4](games/connect4/README.md#the-learned-player) — as does what a run of each actually
produced.

Three invariants make the implementation work, and none of them fails loudly — each produces a
network that trains happily and plays badly. `tests/zero/` pins all three, and the important ones
need no PyTorch at all:

* **The tree is made of paths, not positions.** Children hang off their parent, so a position
  reached by two move orders is two nodes. Keying one flat dict by position looks like an
  optimisation and is a different data structure — 97% of tic-tac-toe positions within five plies
  are reachable more than one way, and re-expanding a shared node resets its statistics and
  re-points its parent. A search built that way gets *worse* with more simulations.
* **One action space over a perspective-relative board.** The planes are always signed from the
  mover's point of view — +1 mine, −1 theirs — so the network learns one player's problem and every
  game teaches it about both seats. A split head halves the data each half sees and invites the two
  halves to disagree about which block is which.
* **The value target is from the mover's own point of view, and a draw is 0.** A position whose
  player went on to win is `+1` *for that position*, whichever player it was.

#### Batching

Batch-1 inference is not *a* cost, it is very nearly the whole cost — a Connect 4 forward pass is
1101µs alone and 111µs amortised in a batch of 64 — and MCTS asks once per simulation. What cannot
be batched is one tree's simulations: they are sequential by construction, each going where the
previous ones' statistics send it, and taking several leaves from one tree needs virtual loss and
*changes what the tree explores*.

So the batching happens somewhere else. The search is a **generator** that yields the position it
needs and is sent back the answer, and self-play advances `--games-in-flight` *separate* games in
lockstep, evaluating their pending positions together. Every game still runs an ordinary sequential
search and gets the tree it would have had alone.

| per generation | one at a time | batched | |
|---|---|---|---|
| Tic-tac-toe, 80 games | 3.43s | **1.16s** | 3.0x |
| Connect 4, 64 games | 75.8s | **15.2s** | 5.0x |

That it is only a speed change is asserted rather than hoped: `tests/zero/test_selfplay.py` plays
the same twelve games at batch 1, 2, 5, 12 and 64 and requires **identical examples**. Grading is
batched too, and there it is free — 22,100 independent positions with no tree involved took 37s one
at a time and take **3.8s** in chunks of `GRADING_CHUNK`.

#### Checkpoints and resuming

A run measured in hours has to survive the machine going away. A checkpoint carries its optimiser
state — Adam keeps a running mean and variance per parameter, and a resume that dropped them would
take its first steps back unmomented — and is written to a temporary file and renamed into place,
because whatever interrupts a long run is just as likely to interrupt the write meant to survive it.

```bash
python3 zero.py --game connect4 train --games 1000 --generations 30 \
    --out models/connect4-best.pt --latest models/connect4-latest.pt \
    --metrics runs/connect4.jsonl --resume --commit-every 2
```

The same command line launches and relaunches: `--resume` continues from `--latest` when that file
exists and starts at generation one when it does not, and `--commit-every` pushes `--latest` and
the metrics to the branch as the run produces them, so a lost machine costs a couple of generations
rather than the run.

**Two files rather than one, and the distinction matters.** `--out` holds the *best* network by the
selection metric, which is what you want to play against; `--latest` holds the most recent one,
which is what a resume must read. Resuming from the best checkpoint silently replays every
generation since the last improvement — the metrics file grows a repeated generation number where
the seam is, which is how this was caught.

**The replay buffer travels beside the checkpoint, not inside it.** `--latest` is the file that
gets committed and pushed every couple of generations, so putting the self-play in it would put
megabytes of games into the history forever to save the one machine that already has them a few
minutes. It goes in a git-ignored `models/connect4-latest.buffer` instead — written every
generation, immediately after the checkpoint, so an interrupted pair leaves the buffer a generation
behind the weights rather than a generation ahead of them.

It matters more than its size suggests. A resume that started with an empty buffer trained its
first generation on a fraction of the usual data and got worse for it, then **spent about three
generations** climbing back — which is an acceptable price for rescuing a run and a trap when it is
not told apart from progress. A Connect 4 run's ladder score "improved" from 0.637 to 0.730 over
six generations after a resume, and the curve was the shape of the buffer rather than of the
player. A run interrupted every few generations never leaves that recovery at all.

Stored as three stacked tensors rather than as the examples themselves: 20,000 Connect 4 positions
are **2.32MB and 0.09s** that way against 7.64MB and 0.44s pickled whole, which is what keeps a
per-generation write from becoming something to think about. Nothing about it is load-bearing — a
missing buffer, one from another game, or one whose planes are the shape an older encoder produced
all log a line and start empty, because a bad buffer costs a few generations and refusing to start
costs the run. A fresh clone finds the pushed checkpoint, no buffer, and resumes exactly as it did
before any of this existed.

#### Metrics

Every generation appends one JSON line to `--metrics`, flushed as it goes, because the run whose
history is most worth having is the one that died at hour three. `plot.py` turns it into a
self-contained page of charts — standard library only, so torch stays the single dependency — and
works mid-run:

```bash
python3 plot.py runs/connect4.jsonl --open
```

The fields are chosen so a flat curve can be *diagnosed*, not just observed. Two are worth knowing
about:

* **`target_entropy`.** Policy loss cannot fall below the entropy of the targets it is fitting, so
  a loss that has flattened *at* that value is a network fitting its targets perfectly and the
  fault is in the search producing them. Tic-tac-toe's `c_puct` being too low looked exactly like
  that.
* **`denormal_weights`.** Weight decay pulls unused weights toward zero and they stall around
  1e-40, which float32 can only represent as a **subnormal**; x86 runs subnormal arithmetic in
  microcode rather than in the vector units. A network 11% denormal trained **6× slower**, and the
  signature defeats the obvious hypotheses — it compounds generation by generation, applies
  uniformly to self-play, grading and gradient steps alike, survives a process restart because the
  weights come back from the checkpoint, and every probe of the machine looks healthy because a
  probe uses a freshly initialised network. `ai.zero.net.flush_denormals` runs after each
  generation's gradient steps and the outputs are asserted identical to the bit.
  (`torch.set_flush_denormal(True)` is *not* an alternative: it sets the CPU flag on the calling
  thread only, and torch runs its intra-op pool on others.)

### Grading a player

Two things can be asked about a player, and they are not the same question:

* **Does it know the game?** `zero.py benchmark` — grade it in every position, including ones no
  sensible game reaches.
* **Can it be beaten?** `zero.py ladder` — play it against a sequence of opponents.

They come apart in one direction, sharply. A player that never blunders *on the path it actually
walks* never arrives at the positions it would get wrong, so it can be wrong in a hundred positions
and still be impossible to beat — the tic-tac-toe network is wrong in 77 and unbeatable.

**Which measure is right is a property of the game, not of the framework.** `SELECTION_METRIC` in
`ai/zero/train.py` holds one choice per game, and a checkpoint records which measure its bar was set
on so a resume cannot silently compare across a change. Tic-tac-toe selects on agreement, because
its ladder saturates once the network plays perfectly and choosing on it means choosing arbitrarily
among ties. Connect 4 selects on the ladder, because its agreement saturates in usefulness: two
networks half a point apart on it scored 0.055 and 0.635 against `minimax:4`. Pick whichever measure
still *moves* where the player actually is.

#### `benchmark`

```bash
python3 zero.py benchmark --player model:models/tictactoe-best.pt --value
python3 zero.py --game connect4 benchmark --player minimax:4 --tier E
```

`ai/oracle.py` asks the player for a move in each of a set of positions and compares it against the
moves that hold the position's value. A move counts as wrong only if it **changes the result** — a
slower win is not a mistake, and neither is an arbitrary choice between moves of equal value.

Where the answers come from is per game. Tic-tac-toe is solved on the spot, in
[all 5,478 positions](games/tictactoe/README.md#grading-against-perfect-play). Connect 4 has ~4.5 ×
10¹² and cannot be enumerated, so its answers were computed once and committed as
[a corpus of 33,300 positions](games/connect4/README.md#the-solved-corpus) in three tiers, selected
with `--tier` and **never averaged together**.

Reports split by seat and by the value of the position, both because an aggregate hides things that
matter. A player can be excellent from one seat and hopeless from the other. And losing positions
are free marks — if every move loses, every move is optimal — while drawn positions, where usually
exactly one move holds, are the ones that actually discriminate.

#### `ladder`

```bash
python3 zero.py ladder --player minimax:9
python3 zero.py --game connect4 ladder --player 'model:models/connect4-latest.pt+mcts:100'
```

A fixed sequence of opponents, a fixed number of games against each — 100 by default, worth about
±0.045. Tic-tac-toe can answer "can it be beaten" exhaustively, since `play_every_line` plays every
line an opponent could take it down; that is exponential in the length of a game, so every other
game gets this affordable substitute.

Every rung is played even after a bad loss, so two runs are always directly comparable — tracking a
player over time is most of what this is for. The summary also **flags a ladder that is not clean**,
reporting both rather than quoting the higher number when a player beats depth 3 but not depth 2.

Two invariants pin the harness. **A deterministic player scores exactly 0.500 against itself**,
which is arithmetic rather than luck: both games of a pair start from the same opening and are
played by the same function, so they run identically and the pair is one win and one loss. And a
**perfect player takes zero losses on every rung**, which is what `minimax:9` on tic-tac-toe is for.

Every player here is deterministic, so two games from the same position are the same game move for
move, and all the variety comes from starting a few random plies in. That makes how those openings
are chosen the whole design:

* **They must be distinct.** `ai/match.py` draws with replacement, which at its defaults gives 30
  distinct openings for 50 pairs of Connect 4 — 40% of a "100-game" match is replayed games, while
  the standard error still divides by 100 and understates itself by about 1.3×. The ladder instead
  uses `ai/oracle.py`'s `openings_at`, which enumerates every distinct position at a ply and
  **raises rather than repeating**.
* **They must be level.** Most openings are already won for somebody — 920 of Connect 4's 1,120 at
  four plies — and a pair from a decided opening is forced to 0.5 as soon as both players convert
  it, which is exactly when the players get good enough to be worth telling apart. Restricting to
  drawn openings turned `minimax:4` vs `minimax:6` from 0.390 ± 0.044 into 0.315 ± 0.041 — noise
  into a result. `--unbalanced` turns that off.

## Development

Nothing to install for the engine or its tests: both are pure standard library. The one optional
extra is PyTorch, for `ai/zero/`.

```bash
python3 -m unittest tests.test_all -v                       # everything
python3 -m unittest tests.test_oracle -v                    # the benchmark, calibrated on known players
python3 -m unittest tests.zero.test_mcts -v                 # PUCT against a perfect evaluator (no torch)
python3 -m unittest tests.zero.test_selfplay -v             # the training targets (no torch)
python3 -m unittest tests.zero.test_net -v                  # the network and its checkpoints (torch)
```

Each game's own suites are listed on its page —
[chess](games/chess/README.md#tests),
[Connect 4](games/connect4/README.md#tests),
[tic-tac-toe](games/tictactoe/README.md#tests) — and each includes a `test_conformance.py` running
it against the shared `GameState` contract.

`tests/zero/test_mcts.py` and `tests/zero/test_selfplay.py` deliberately need no PyTorch. The parts
of a learned player most likely to be wrong — the tree and the training targets — are the parts
that can be checked without a single trained weight, and a bug in either looks exactly like a
network that has not learned yet. Hand the search a perfect evaluator built from the tic-tac-toe
oracle and it must play perfectly; hand it one that knows nothing and it must *improve* with more
simulations.
