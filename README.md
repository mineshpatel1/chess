# Mildred

Game engines written from scratch in pure Python — bitboard move generation, alpha-beta search,
and a UCI interface for the chess. No third-party runtime dependencies.

```
8 [♜][♞][♝][♛][♚][♝][♞][♜]
7 [♟][♟][♟][♟][♟][♟][♟][♟]
6 [ ][ ][ ][ ][ ][ ][ ][ ]     [ ][ ][ ][ ][ ][ ][ ]
5 [ ][ ][ ][ ][ ][ ][ ][ ]     [ ][ ][ ][ ][ ][ ][ ]     [✕][1][○]
4 [ ][ ][ ][ ][ ][ ][ ][ ]     [ ][ ][ ][○][ ][ ][ ]     [3][✕][5]
3 [ ][ ][ ][ ][ ][ ][ ][ ]     [ ][ ][ ][●][ ][ ][ ]     [6][7][○]
2 [♙][♙][♙][♙][♙][♙][♙][♙]     [ ][ ][○][●][ ][ ][ ]
1 [♖][♘][♗][♕][♔][♗][♘][♖]     [ ][●][●][○][ ][ ][ ]
   A  B  C  D  E  F  G  H       0  1  2  3  4  5  6
```

Chess is why the project exists. The other two are the evidence that the search underneath it is
not secretly chess — all three games share one search, one move enumerator and one conformance
suite, and `ai/` contains no chess at all.

Connect 4 makes that case by being different from chess. Tic-tac-toe makes it by being **solved**:
nine cells fit inside a nine-ply search, so the same alpha-beta that plays chess approximately
plays this one perfectly, and `tests/tictactoe/test_perfect_play.py` checks it against an
independent solver in every one of the game's 5,478 positions. It is also the smallest thing that
can satisfy the contract — three files, no bitboard module, and a move that is just the number of
a cell.

Being solved makes it the right game to learn, too. There is an **AlphaZero implementation** in
`ai/zero/` that trains a network from nothing but self-play, and because the answer is computable
it can be graded against perfect play in every position rather than against an opponent. See
[Learning to play](#learning-to-play).

## Quick start

Requires Python 3.7+. Nothing to install.

```bash
python3 -m unittest tests.test_all -v     # run the test suite
python3 play.py                           # play any of the three games in the terminal
python3 -m games.chess.uci                # start the UCI engine on stdin
python3 zero.py benchmark --player minimax:9   # grade a player against perfect play
python3 zero.py --game connect4 ladder --player minimax:4   # place it against a set of opponents
```

The learned player is the one exception to "nothing to install", and it is opt-in:

```bash
pip install -r requirements-zero.txt      # PyTorch, and only for ai/zero/
python3 zero.py train                     # train a tic-tac-toe network from scratch
```

Talking to it directly:

```
$ python3 -m games.chess.uci
uci
id name Mildred
id author Nesh Patel
option name skill type spin default 3 min 0 max 5
uciok
position startpos moves e2e4 e7e5
go
bestmove g1f3
d
<prints the board>
quit
```

## Using it in a chess GUI

Mildred speaks [UCI](https://en.wikipedia.org/wiki/Universal_Chess_Interface), so it can be
registered as an engine in Arena, CuteChess, Scid or any other UCI-compatible GUI. Point the
GUI at a launcher script:

```bash
#!/bin/sh
cd /path/to/chess && exec python3 -m games.chess.uci
```

### Options

| Option  | Type | Default | Range | Meaning |
|---------|------|---------|-------|---------|
| `skill` | spin | 3       | 0–5   | Search depth in plies. `0` plays a uniformly random legal move. |

Search time grows steeply with depth — see the benchmarks below.

## How it works

| Module | What is in it |
|---|---|
| `games/base.py` | The `GameState` contract — the only thing the search knows about a game |
| `games/chess/bitboard.py` | 64-bit board representation, precomputed attack/ray tables |
| `games/chess/board.py` | `ChessBoard`: position, move generation, legality, FEN, draws |
| `games/chess/evaluation.py` | Material and piece-square evaluation |
| `games/chess/move.py`, `piece.py`, `square.py` | Value types |
| `games/chess/uci/` | Mildred's own UCI server, plus an async UCI *client* for driving others |
| `games/chess/run_engine.py` | Playing Mildred against a third-party engine |
| `games/connect4/constants.py` | The board's shape, and every mask derived from it |
| `games/connect4/bitboard.py` | The carry that drops a disc, and the shifts that find four in a row |
| `games/connect4/board.py` | `Connect4`: two integers, move generation, make/unmake, win detection |
| `games/connect4/evaluation.py` | Open threes, weighted by direction and by whether they are live |
| `games/tictactoe/constants.py` | Nine cells, eight winning lines, and the order moves are tried in |
| `games/tictactoe/board.py` | `TicTacToe`: two 9-bit integers, and a `SOLVED_DEPTH` of nine |
| `games/tictactoe/evaluation.py` | Open twos, which matter only below the depth the game is solved at |
| `games/tictactoe/encoding.py` | What a network sees: one signed plane, nine shared actions |
| `ai/search.py` | Negamax with alpha-beta pruning, and a random mover |
| `ai/perft.py`, `ai/simulate.py` | Move enumeration, and playing two move-choosers off |
| `ai/match.py` | Playing them off a few hundred times, which is how an evaluation is judged |
| `ai/ladder.py` | Placing a player against a fixed sequence of opponents — can it be beaten |
| `ai/oracle.py` | Exact play, and grading any player against it over a set of positions |
| `ai/corpus.py` | Reading positions whose exact value was computed once and written down |
| `ai/generate.py` | Choosing those positions, and getting the answers from an external solver |
| `ai/corpora/connect4.txt` | 33,300 Connect 4 positions, every legal move in each valued exactly |
| `ai/players.py` | Naming a player — `minimax:9`, `model:best.pt+mcts:200` — in one place |
| `ai/zero/mcts.py` | PUCT search over a tree of paths, with no rollouts |
| `ai/zero/net.py` | Two trunks — dense for tic-tac-toe, a residual tower for Connect 4 |
| `ai/zero/selfplay.py` | Games against itself, and the targets they produce |
| `ai/zero/train.py` | The generation loop, graded against the oracle as it goes |
| `ai/zero/metrics.py` | One JSON line per generation, flushed as it goes, so a dead run keeps its history |
| `games/connect4/encoding.py` | What a network sees: two planes of 6x7, seven shared actions |
| `play.py` | A terminal front end for any game in the registry |
| `zero.py` | Training, grading and comparing learned players |
| `bench.py` | Timing the exact solver, and finding how far back into a game it reaches |
| `plot.py` | A training run's metrics as a self-contained page of charts, no matplotlib |
| `tests/conformance.py` | The tests every game must pass |
| `tests/chess/` | Rules, replay/undo, perft, and search scores, all driven by chess positions |
| `tests/connect4/` | The same, plus the exhaustive win-detection oracle and the eval properties |
| `tests/connect4/solved.py` | 280 positions solved exactly and written down, so the solver cannot drift |
| `tests/tictactoe/` | The same, plus the proof that the engine plays the game perfectly |

In chess, positions are held as twelve 64-bit integers (one per piece type per colour) plus
occupancy masks. Sliding attacks are generated by masking precomputed rays against blockers;
knight, king and pawn moves come from lookup tables built at import time.

Undo is cheap: `_BoardState` snapshots the raw integers rather than copying the board, so
`make_move`/`unmake_move` is the hot path for search rather than a bottleneck.

Connect 4 is two integers, one per player, over a board with a **sentinel row**: a cell is
`column * STRIDE + row` where `STRIDE` is one *more* than the number of rows, so each column
carries a spare cell above it that is never occupied. That extra bit is what makes the rest of
the engine simple. Move generation is a single addition — `(occupied + BOTTOM_ROW) & FULL_BOARD`
rings a carry up every column at once and it settles on the cell a disc would land in, with a
full column's carry absorbed by its sentinel — so there is no `height[]` array to keep in step.
Win detection is four shift chains, one per direction, halving the problem each time: `pos &
(pos >> d)` marks adjacent pairs, and the same again at `2d` marks runs of four. Neither can wrap
from the top of one column into the bottom of the next, because the sentinel is in the way.

Tic-tac-toe needs none of that, which is the point of it being here. Nine cells have exactly eight
winning lines, so they are enumerated once at import and a win is `marks & line == line` — no
padding, no shifts, no bitboard module. What it has instead is `SOLVED_DEPTH = 9`: the whole game
tree fits inside a search, so `play.py` searches all of it by default and the engine plays
perfectly rather than well. A full solve from an empty board takes about 30ms, which is why this
needs no transposition table, no iterative deepening and no memoisation — the shared search is
already fast enough to be exact.

Its move ordering is derived rather than declared: cells are tried in order of how many winning
lines run through them, which puts the centre first (four lines), then the corners (three), then
the edges (two). Since every opening move is theoretically drawn, the search has nothing to choose
between them and takes the first one generated — so that ordering is also the entire opening book.

## Adding a game

The search in `ai/` knows nothing about chess. It asks a game for its legal moves, plays them,
unplays them, and asks how a finished position finished — so anything that can answer those
questions can be searched. A move can be whatever suits the game; the search only ever hands
one back.

1. **Implement `GameState`** (`games/base.py`) in `games/<name>/`. Four methods —
   `legal_moves`, `make_move`, `unmake_move`, `copy` — plus `outcome_without_moves`, which
   says what it means to have nothing to play. In chess that is checkmate or stalemate; in a
   game that fills up, it is a draw.
2. **Override `outcome`** if the game can be won while moves remain — a line of three, a line
   of four. It is checked at every node, so keep it cheap. It defaults to `None`, which is
   right for games that end by running out of moves, so chess does not implement it at all.
   Connect 4 is the worked example: it tests only the board of the player who just moved,
   because only that player can have just won. `ai/perft.py` honours it too, so that a decided
   game is counted as having no continuations rather than being played on through.
3. **Write an evaluation** returning a score that is positive for the player to move. The
   search is negamax, so every score is read from the point of view of whoever is on move.
   Name it as the class's `DEFAULT_EVAL`, and keep it well inside `ai.search.MATE` or a good
   position becomes indistinguishable from a won one. Justify it with `ai/match.py` rather
   than by eye — terms that feel obviously right frequently are not.
4. **Generate moves in a useful order.** `legal_moves` may return them in any order, and the
   search takes them in that order and breaks ties by it, so ordering is the game's business
   and it is most of what makes alpha-beta pay. Connect 4 offers the centre column first,
   which is worth a factor of nineteen at depth 6.
5. **Set `PARALLEL_ROOT`** if the game branches widely enough that splitting the root across a
   process pool beats starting one. Chess does; a game with seven or nine moves does not.
6. **Override `parse_move`** if a person typing a move at `play.py` deserves better than the
   default, which matches what they typed against `str(move)`. Optional — all three games have a
   working default and override it only to say *why* a move was refused.
7. **Set `SOLVED_DEPTH`** if the whole game fits inside a search, as tic-tac-toe's nine plies do.
   `play.py` uses it as its default depth, so a solved game arrives at the prompt playing
   perfectly instead of merely well. Leave it `None` otherwise — chess and Connect 4 do.
8. **Register it** in `games/__init__.py` and **subclass `GameConformanceTests`**
   (`tests/conformance.py`), filling in its three hooks. That suite is what catches the
   mistakes that do not look like mistakes — an undo that restores almost everything, a copy
   that shares state with its original.

Nothing in `ai/` should need to change, and so far nothing has needed to twice over. Adding
Connect 4 changed one thing there — `perft` now asks whether a position is decided, which chess
never needed because a mated position generates no moves. Adding tic-tac-toe changed nothing in
`ai/` at all.

It did change `games/base.py` twice, though, and honestly rather than not:

* `SOLVED_DEPTH` was added, which is the stage 7 above — a new optional attribute that games
  without one never notice.
* `GameState.result` had a bug, and tic-tac-toe found it immediately. It asked
  `any(self.legal_moves)`, which tests whether a move is *truthy* rather than whether one exists.
  Both small games number their moves from zero, so move 0 is legal and falsy: Connect 4 called a
  drawn game with six cells free in column 0, and tic-tac-toe did it whenever the last empty cell
  was the top-left corner, which is about one game in nine. `games.base.has_moves` asks the
  question properly now, and `tests/conformance.py` grew a test that every game runs — one that
  walks by `legal_moves` rather than by `is_game_over`, because a walk that trusts the thing
  under test stops one move before the position that breaks it. That is exactly how it survived
  Connect 4 being added.

The audit that followed found the same shape of mistake once more, in `ChessBoard.copy`. It
rebuilt the board from its FEN, which carries castling rights, the en passant square and both
clocks — but not how many times a position has occurred. So a copy of a game drawn by threefold
repetition had the **same `signature` as the original and was still running**, which is precisely
what `GameState.signature` promises cannot happen. Nothing enabled `track_repetitions` outside one
test, so it never bit; but copies are not incidental — `alpha_beta` takes one per root move and
`ai/match.py` plays every game on one.

Both bugs are the same failure, and it is worth naming because it is not a coding mistake:

> **The test's notion of "the same" was weaker than the behaviour that mattered.**

`any(moves)` asked whether a move was truthy when the question was whether one existed.
`test_copy_is_equal_and_independent` compared a copy on `_identity`, which is built from
`signature`, so anything `signature` omitted was invisible to it by construction. In both cases
the suite was thorough about the wrong equality.

The fix in both cases is to compare what a caller can actually *observe* rather than what the
implementation happens to expose, which is what `test_a_copy_is_interchangeable_with_its_original`
now does — moves, both halves of the result, and the turn. That also gives `copy` a real
definition of what it may leave behind: anything invisible to that test. `move_history` qualifies,
being read only by `pgn_uci`; the repetition history never did.

## Learning to play

`ai/zero/` is an AlphaZero implementation: a network learns to play tic-tac-toe from nothing but
games against itself. No opening book, no evaluation function, no opponent — the only fact from
outside the process is who won.

It needs PyTorch, and it is the only thing here that does. The engine, `play.py` and the whole
test suite run on the standard library; the learning tests skip themselves when torch is absent.

```bash
pip install -r requirements-zero.txt
python3 zero.py train
python3 zero.py benchmark --player model:models/tictactoe-best.pt
python3 zero.py benchmark --player model:models/tictactoe-best.pt+mcts:200
python3 zero.py match --a model:models/tictactoe-best.pt+mcts:200 --b minimax:9 --games 200
python3 play.py                    # choose "Model" and play it yourself
```

A trained checkpoint is committed, so all of that works immediately after cloning.

### Naming a player

`ai/players.py` turns a string into the `state -> move` callable every harness here already
expects, so the terminal, the benchmark and the match harness all mean the same thing by an
opponent:

| Spec | Player |
|---|---|
| `random` | a uniformly random legal move |
| `minimax:9` | alpha-beta, which at depth 9 is perfect |
| `model:PATH` | the network's policy, **with no search at all** |
| `model:PATH+mcts:200` | the same network, thinking 200 simulations ahead |

The last two are the reason it exists. Raw intuition and intuition-plus-search are one clause
apart, so comparing them is a change of argument rather than of code — and both go through the
same benchmark as the classical players.

### Grading against perfect play

Because tic-tac-toe is solved, a player can be graded against the answer rather than against an
opponent. `ai/oracle.py` walks **every one of the 5,478 reachable positions**, asks the player for
a move, and compares it with the set of moves that hold the position's value.

That is a much harder test than playing matches, and a more useful one. A player only ever meets
its own blunders through an opponent willing to punish them, and losing lines are precisely the
ones a decent opponent never steers into. Walking every position has no opponent in it and no
sampling either.

The report splits by seat, and that split is the point:

```
$ python3 zero.py benchmark --player model:models/tictactoe-best.pt --value

  overall       97.9% optimal (4423/4520), 97 blunders, 0.0228 mean value lost
  as first      97.3% optimal (2358/2423), 65 blunders, 0.0289 mean value lost
  as second     98.5% optimal (2065/2097), 32 blunders, 0.0157 mean value lost
  by ply       0:100%  1:100%  2:100%  3:98%  4:95%  5:99%  6:98%  7:98%  8:100%
  value head   0.1910 mean squared error vs truth
```

A player can be excellent from one seat and hopeless from the other, and an aggregate hides it
behind an average. That is not hypothetical: it is what the 2021 attempt in this repository's
history actually did, and it went unnoticed for months.

### Two different questions

There are two things you can ask about a player, and they are not the same question:

* **Does it know the game?** Grade it in every position, including ones no sensible game reaches.
* **Can it be beaten?** Play it against every line an opponent could take it down.

They come apart in one direction, and sharply. A player that never blunders *on the path it
actually walks* never arrives at the positions it would get wrong — so it can be wrong in a
hundred positions and still be impossible to beat. `zero.py benchmark` reports both.

### Reproducing the committed checkpoint

```bash
python3 zero.py train
```

That is the whole command — the CLI defaults *are* the recipe. Written out in full, and every
argument here is also the default:

```bash
python3 zero.py train \
    --generations 400 \        # 45 min on four CPU threads
    --games 80 \               # self-play games per generation
    --simulations 50 \         # MCTS simulations per move
    --steps 60 \               # gradient steps per generation
    --seed 1 \
    --out models/tictactoe-best.pt
```

The rest lives in `ai/zero/train.py` and `ai/zero/selfplay.py`, and each one has its measurement
recorded next to it:

| Parameter | Value | Why |
|---|---|---|
| Network | 9 → 64 → 64 → (9 policy, 1 value) | One signed plane beat two binary ones by 1.2 points |
| `SELF_PLAY_EXPLORATION` | **5.0** | c_puct 1.5/3.0/5.0/8.0 → 92.8/95.4/95.8/94.7% |
| `SIMULATIONS` | **50** | 50→200 buys 1.0 point for 3.1× the time |
| `TEMPERATURE_MOVES` | 30 (all nine plies) | Cutting to 3 plies cost 10 points |
| `OPENING_PLIES` | **0** | Random starts were covering for c_puct being too low |
| `SYMMETRIES` | **False** | No benefit, and it is knowledge the network should not be given |
| `DIRICHLET_EPSILON` / `ALPHA` | 0.25 / 1.0 | Root noise, self-play only — never at evaluation |
| Optimiser | Adam, lr 1e-3, decay 1e-4, batch 128 | |
| `BUFFER_SIZE` | 20,000 positions | |

Training is deterministic given `--seed`, so the command above reproduces the committed weights.

Grading it afterwards is two more commands:

```bash
python3 zero.py benchmark --player model:models/tictactoe-best.pt            # the network alone
python3 zero.py benchmark --player model:models/tictactoe-best.pt+mcts:50    # with search
```

### Results

The committed checkpoint: **45 minutes on four CPU threads**, 25KB of weights. Purely on-policy —
no random openings, no symmetry augmentation, no supervision, nothing but self-play.

**Can it be beaten? No — not even with the search switched off.**

| Player | vs *any* opponent, as first | vs *any* opponent, as second |
|---|---|---|
| **Raw policy, no search** | 107W / 4D — **0 losses** | 432W / 125D — **0 losses** |

Every line, both seats, no losses. The network alone holds a draw against perfect play and beats
everything that misplays — which is the whole of what tic-tac-toe asks of a player.

**Does it know the game?** Less completely, and that is where the remaining gap is:

| Player | Overall | Wrong (of 4,520) |
|---|---|---|
| Raw policy, no search | 98.30% | 77 |
| + 25 simulations | 99.69% | 14 |
| + 50 simulations | 99.87% | 6 |
| + 200 simulations | **99.98%** | **1** |
| `minimax:9` (perfect by construction) | 100% | 0 |

A move counts as wrong only if it **changes the result** — a slower win is not a mistake, and
neither is an arbitrary choice between moves of equal value. `solve()` returns 1/0/−1 with no
depth term, so a mate in three scores the same as a mate in one.

The raw-policy mistakes are all mid-game, and nearly all throw away a draw rather than a win.
Every one is in a position the model never reaches when it is the one playing — which is exactly
why the table above reads "unbeaten" while this one does not read 100%.

The last two columns are the same model counted two ways, and the gap between them is a trap
worth naming. Tic-tac-toe's 4,520 decision positions collapse to **627 distinct boards** once the
eight symmetries are folded together, so "31 wrong" and "9 wrong" describe the same network. Any
comparison against another implementation has to agree on which is being counted before the
numbers mean anything.

Being able to run the network with search switched off is why `simulations` is a parameter rather
than two implementations. A network that only plays well with 500 simulations has learned to be a
useful prior for a search doing the real work; one that is unbeatable at zero simulations has
learned the game itself.

### What it took to get right

The three things that make this work are all things an earlier attempt got wrong, and none of
them fails loudly — each produces a network that trains happily and plays badly.

**The tree is made of paths, not positions.** Children hang off their parent, so a position
reached by two move orders is two nodes. Keying one flat dict by position looks like an
optimisation and is a different data structure: 97% of tic-tac-toe positions within five plies are
reachable more than one way, and re-expanding a shared node resets its statistics and re-points
its parent. A search built that way gets *worse* with more simulations.

**One action space, over a perspective-relative board.** The plane is always signed from the
mover's point of view — +1 mine, −1 theirs — so the network learns one player's problem and every
game teaches it about both seats. Nine actions, not nine per player: a split head halves the data
each half sees and invites the two halves to disagree about which block is which.

**The value target is from the mover's own point of view, and a draw is 0.** A position whose
player went on to win is `+1` *for that position*, whichever player it was. Most tic-tac-toe games
are drawn, so most examples must say so.

`tests/zero/` pins all three, and the important ones need no PyTorch at all: hand the search a
**perfect evaluator** built from the oracle and it must play perfectly; hand it one that knows
nothing and it must improve with more simulations.

| Evaluator | 10 sims | 50 sims | 200 sims | 800 sims |
|---|---|---|---|---|
| Perfect (the oracle) | 100% | 100% | 100% | 100% |
| Ignorant (flat priors, zero value) | 89.7% | 97.1% | 99.3% | 100% |

The first row says the search uses knowledge correctly. The second says it generates knowledge on
its own — and that it climbs, which is the property the 2021 version lacked. Both are checked
without a single trained weight, which is exactly the test whose absence let that version stay
broken.

### The tuning, and a wrong turn worth recording

The first version of this reached only 80.3% on-policy. Forcing a share of self-play games to
start from a random position took it to 96.8%, which looked like the single largest improvement in
the implementation.

It was not an improvement. It was covering for two exploration parameters being set wrong, and
finding that out took measuring each one:

| Change | On-policy agreement |
|---|---|
| Starting point | 80.3% |
| Temperature over the whole game, not the first 3 plies | 93.7% |
| `c_puct` 1.5 → 5.0 | **97.5%** |
| *(random openings, for comparison)* | *96.8%* |

`c_puct` alone, measured over 90 generations at 50 simulations, is an inverted U with a clear peak:

| c_puct | 1.5 | 3.0 | **5.0** | 8.0 |
|---|---|---|---|---|
| Agreement | 92.83% | 95.38% | **95.84%** | 94.65% |

At 1.5 the visit counts were concentrating before the alternatives had been checked, so the
network was fitting targets that were confidently slightly wrong — and its policy loss sat at the
entropy floor of those targets, which is what "fitting them perfectly" looks like. **Random
openings are now off by default.** A correct AlphaZero reaches the state space through PUCT and
root noise; needing to force random starts on top is a sign that one of those is mistuned.

Simulations during self-play turned out to matter far less than the cost suggests:

| Simulations | 25 | **50** | 100 | 150 | 200 |
|---|---|---|---|---|---|
| Agreement | 87.4% | **92.8%** | 91.7% | 93.7% | 93.8% |
| Time | 276s | **337s** | 590s | 804s | 1044s |

Going 50 → 200 costs 3.1× the time for 1.0 point, so the default is 50. More search does sharpen
the visit counts (target entropy 0.92 → 0.73) and does narrow self-play coverage by ~20% (546 →
434 distinct positions), but the sharper targets more than pay for the lost breadth — the effect
is real and the sign is the opposite of a problem.

Every one of these is a parameter of `train()`, which is the only reason any of it could be
measured rather than argued about.

## Benchmarks

Move generation, measured on a 4-core container with CPython 3.11:

```
perft(4) from the starting position: 197,281 nodes at ~110,000 nodes/sec
```

### Perft correctness

Verified against the [reference results](https://www.chessprogramming.org/Perft_Results).

| Position | Depth | Expected | Status |
|---|---|---|---|
| Starting position | 4 | 197,281 | ✅ |
| Kiwipete | 3 | 97,862 | ✅ |
| Position 3 | 4 | 43,238 | ✅ |
| Position 4 | 3 | 9,467 | ✅ |
| Position 5 | 3 | 62,379 | ✅ |

All five run as part of the test suite (`tests/chess/test_permutations.py`), so move generation
is pinned to the reference counts rather than merely known to have matched them once.

Connect 4 has no published table, so its counts are *derived* rather than looked up — and the
tree is unconstrained long enough to make that easy. A column takes six discs and the first
player's fourth disc arrives at ply 7, so nothing can fill or win inside six plies:

| Depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Nodes | 7 | 49 | 343 | 2,401 | 16,807 | 117,649 | **823,536** | **5,673,234** |

The first six are exactly 7ᵈ. Depth 7 is `7⁷ − 7`, the seven prefixes that stack a single column
and then have one move fewer. Depth 8 is where wins start truncating lines and hand derivation
gives out; it runs at ~640,000 nodes/sec, and is left out of the suite at ~9s.

Tic-tac-toe is small enough to have been counted exhaustively long ago, so it is the only game
besides chess with a genuine external oracle — and unlike chess's, the whole table fits:

| Depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Nodes | 9 | 72 | 504 | 3,024 | 15,120 | 54,720 | 148,176 | 200,448 | 127,872 |

The first five are the falling factorial `9!/(9−d)!`: a win needs three marks, so the first
player's third at ply 5 is the earliest a game can end, and nothing before it constrains anything.
Depth 6 is the first count that is not — 5,760 of the 60,480 six-move sequences are games somebody
had already won.

Stronger still is the census, which perft cannot give because counting sequences cannot tell a won
game from a drawn one. Every one of the 255,168 playable games is played out in the suite and its
result recorded, against the published totals:

| First player wins | Second player wins | Draws | Total | Reachable positions |
|---|---|---|---|---|
| 131,184 | 77,904 | 46,080 | **255,168** | **5,478** |

### Perfect play

The whole game is searchable, so tic-tac-toe is tested against the answer rather than against a
proxy for it. `tests/tictactoe/test_perfect_play.py` writes an independent memoised minimax —
no alpha-beta, no move ordering, no negamax — and checks the engine against it three ways:

| Claim | How it is checked |
|---|---|
| The search plays optimally | Its move is one the solver would play, in all 5,478 positions |
| The pruning is sound | Alpha-beta's score agrees with unpruned minimax, in all 5,478 positions |
| The engine cannot be beaten | Played against *every* opponent line, as first player and as second |

All of it runs in about 1.5 seconds, and the last row is the one a person cares about: there is no
sequence of moves that beats it from either side.

### How far back into Connect 4 the exact solver reaches

Tic-tac-toe could be graded against the answer because all 5,478 positions fit in a loop. Connect 4
has about 4.5 × 10¹² and cannot be enumerated, so grading a player there means **sampling positions
and solving each one exactly**. How early in a game those positions can be drawn from is decided by
nothing but how fast the solver is, which is why it got a project of its own.

It was full-width negamax with a string-keyed memo and no pruning. It is now alpha-beta over a
transposition table carrying bound flags, an immediate-win shortcut, table-driven move ordering,
and mirror-symmetry sharing. Time to solve one position from a cold table, median over sampled
positions at each ply:

| Discs on the board | 24 | 22 | 20 | 18 | 16 | 14 |
|---|---|---|---|---|---|---|
| Before | 0.29s | ~1.3s | *did not finish* | *did not finish* | *did not finish* | *did not finish* |
| After | 0.000s | 0.010s | 0.088s | 0.054s | 1.85s | 22.8s |
| Worst seen, after | 0.34s | 0.42s | 1.54s | 3.42s | 21.1s | 69.9s |

**The frontier moved eight plies**, from 24 to 16. Two hooks on `GameState` carry it generically,
both defaulting to something correct and slow so tic-tac-toe and chess are untouched:

| Hook | What Connect 4 supplies |
|---|---|
| `solver_key` | `discs[turn] + occupied` — the whole position as one integer, exact rather than a hash |
| `canonical_key` | the same, or its left-right mirror, whichever is smaller |
| `winning_moves` | `completions(mine) & drops(occupied)` — can I win right now, in bit operations |

The mirror is the part that could go wrong in silence, so the design removes the opportunity.
**Value bounds are stored under `canonical_key`** — sound, because a reflected board is worth
exactly what the original is. **Move-ordering hints are stored under `solver_key`**, in a separate
dictionary, and are never mirrored: the best move in a reflected position is the *reflected* move.
No move can come out of a mirrored entry because there is no mirrored entry to read one from.

`winning_moves` is worth its own note, because the profiler put about half the running time in that
one question, asked at every node. Answering it by playing all seven columns and testing each board
costs 18.8µs; `completions` marks every cell that would finish a line of four and intersects it
with where a disc would land, for 4.5µs.

That the answers did not change is checked rather than claimed. `tests/connect4/solved.py` pins 280
positions — value *and* full optimal-move set, forty at each even ply from 22 to 34 — generated by
the unoptimised solver before any of this landed:

```bash
python3 bench.py corpus                       # re-solve the pinned set, timed: 280 positions in 3.9s
python3 bench.py frontier --plies 20 18 16    # what fresh positions cost now
python3 bench.py verify                       # re-solve the corpus ourselves, as deep as we reach
```

Two more checks sit beside it, and neither is the solver grading itself. **Mirror invariance** —
`solve(position) == solve(mirror(position))` — needs no oracle at all, holds everywhere, and is
exactly the fault the symmetry sharing could introduce. And the **published solution**: Connect 4
was solved in 1988 by Allis and independently by Allen, so facts about it exist outside this
repository, in the same way chess's perft counts do.

A caveat on reading the table: it is the cost of solving one position from a *cold* table, because
that is what a sampled benchmark does. Positions that share a table are far cheaper — which is also
why a corpus is affordable much deeper than interactive use is. Pinning a few hundred ply-14
positions is a couple of hours run once; asking for one at the prompt is not.

### Grading a Connect 4 player against the answer

Tic-tac-toe is graded by walking all 5,478 of its positions and solving each one on the spot.
Connect 4 has ~4.5 × 10¹², so the answers are computed once, ahead of time, and committed:
**`ai/corpora/connect4.txt`, 33,300 positions, every legal move in each of them valued exactly.**

Every move, not just the best ones. `benchmark` scores a player by how much value its move gave
away, so a file holding only the optimal set could not tell a thrown-away draw from a thrown-away
win.

Three tiers, by how the positions were chosen, and **they are never averaged together**:

| Tier | Plies | How chosen | Count |
|---|---|---|---|
| `E` | 0–6 | **every** distinct position — enumerated, not sampled | 22,100 |
| `R` | 7–34 | seeded random play, 200 per ply | 5,600 |
| `P` | 7–34 | games between alpha-beta players, deviating 15% of the time | 5,600 |

The opening is enumerated because it fits: the game opens narrowly, at 7, 49, 238, 1,120, 4,263 and
16,422 distinct positions per ply, so plies 0–6 are four times the whole tic-tac-toe state space and
still trivial. There is no sampling to be biased, and it stops at six because ply seven is the
earliest a game can end. It is also the tier that matters most — Connect 4 is a first-player win,
and the opening is where that win is kept or thrown away.

`R` and `P` stay apart because they measure different things. Random play reaches positions no
sensible game visits; play between decent players never asks a player to recover from a bad
position, which is where blunders live. Every ply is sampled, odd and even, because ply parity *is*
whose turn it is — a corpus of even plies only would ask the first player everything and the second
player nothing.

```
$ python3 zero.py --game connect4 benchmark --player minimax:4
```

| | opening (22,100) | random play (5,600) | real play (5,600) |
|---|---|---|---|
| `random` | 54.1% | 59.2% | 72.1% |
| `minimax:4` | **79.1%** | **95.5%** | **91.5%** |

Two things in that table justify the whole design. The opening is *much* harder than anywhere else
— depth 4 scores 79.1% there against 95.5% on deep random positions — which is exactly the region a
sampled-only corpus would have covered worst. And the tiers disagree about which is harder
depending on who is asked: `P` is easier than `R` for the random player and harder for `minimax:4`.
Averaging them would have produced a single number that moved for reasons having nothing to do with
the player.

That inversion has a cause, and it is why `Report` now splits by the value of the position:

| `minimax:4` on | winning positions | drawn positions | losing positions |
|---|---|---|---|
| opening | 76.1% | **58.6%** | 100% |
| random play | 95.1% | **78.1%** | 100% |
| real play | 86.5% | **88.8%** | 100% |

**Losing positions are free marks** — if every move loses, every move is optimal — and real play
reaches far more of them (1,877 of 5,600) than random play does (1,406). Drawn positions are the
opposite: usually exactly one move holds and the rest lose, so that column is the one that actually
discriminates. An aggregate over a set that happens to be mostly decided says more about the
sampling than about the player.

#### Where the answers came from, and why you can believe them

Our solver reaches ply 16 and the corpus needs ply 0, which no amount of optimisation closes. The
values are computed by [Pascal Pons' Connect 4 solver](https://github.com/PascalPons/connect4) with
its 32MB opening book. It is AGPL and is **not vendored** — it is fetched and built into
`third-party-engines/`, which is gitignored, and only the numbers are committed:

```bash
mkdir -p third-party-engines/connect4 && cd third-party-engines/connect4
for f in Makefile main.cpp Solver.cpp Solver.hpp Position.hpp \
         TranspositionTable.hpp MoveSorter.hpp OpeningBook.hpp; do
  curl -O "https://raw.githubusercontent.com/PascalPons/connect4/master/$f"
done
make && curl -L -o 7x6.book \
  "https://github.com/PascalPons/connect4/releases/download/book/7x6.book"

python3 -c "from ai.generate import build; build('ai/corpora/connect4.txt', count=200, seed=0)"
```

Only the *sign* of its score is kept. Pons scores by distance to the end of the game, which would
have to be decoded correctly to be used and would be silently wrong if it were not — and the sign is
all the {-1, 0, 1} convention here needs. It also matches how the benchmark grades: a slower win is
not a mistake.

So the corpus is one external program's opinion, and it is checked four ways, none of which is that
program agreeing with itself:

| Check | What it rests on | Result |
|---|---|---|
| **The game tree** | The `E` tier is closed under it — a position is worth its best reply, negated, and both are in the file | 39,746 parent/child pairs, **0 inconsistent** |
| **The published solution** | Allis and Allen, 1988, independently | Openings read `-1 -1 0 1 0 -1 -1`; centre is the unique win |
| **Our own solver** | Shares no code — Python negamax over a padded bitboard vs C++ with a book | **7,600 positions** re-solved at plies 16–34, **0 disagreements** |
| **Mirror symmetry** | A reflected board is the same game | Every `E` entry matches its reflection |

The first is the strongest and costs nothing: tens of thousands of entries checked against tens of
thousands of others, with no solver involved. A column mapped the wrong way, a dropped sign, a
mislabelled tier — none survive it. It runs in the suite, in three seconds, and needs no download.

Plies 7–14 have no independent check, which is stated rather than papered over. They are bracketed
by exact checks at both ends and produced by the pipeline the `E` tier proves out.

One subtlety worth recording, because it was nearly a bug. Connect 4 is mirror-symmetric, so a
column-orientation mismatch between the two solvers would leave every *value* right and every
*optimal move set* reversed. It turned out not to be a hazard at all — a consistent relabelling is
harmless, because the mirror cancels — but an *inconsistent* one is fatal, and the check has teeth:
mapping the input and not the output breaks 219 of the 280 pinned positions, exactly the ones whose
optimal set is asymmetric.

### Can it be beaten? The match ladder

Grading against the corpus asks whether a player **knows** the game. It is a different question
from whether the player **can be beaten**, and the two come apart sharply in one direction: the
tic-tac-toe network is wrong in 77 positions and is still unbeatable, because a player that does
not blunder on the path it walks never arrives at the positions it would get wrong.

Tic-tac-toe answers the second question exhaustively — `play_every_line` plays every line an
opponent could take it down. That is exponential in the length of a game, so Connect 4 gets the
affordable substitute: a fixed sequence of opponents, a fixed number of games against each.

```
$ python3 zero.py --game connect4 ladder --player minimax:4
```

| Challenger | `random` | `minimax:1` | `:2` | `:3` | `:4` | `:5` | `:6` | Highest beaten |
|---|---|---|---|---|---|---|---|---|
| `random` | 0.460 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **none** |
| `minimax:2` | 1.000 | 0.925 | 0.500 | 0.535 | 0.510 | 0.275 | 0.390 | **`minimax:1`** |
| `minimax:4` | 1.000 | 0.935 | 0.490 | 0.640 | 0.500 | 0.405 | 0.315 | **`minimax:3`** |

100 games per rung, about two minutes for the lot. Every rung is played even after a bad loss, so
two runs are always directly comparable — tracking a player over time is most of what this is for.

**A deterministic player scores exactly 0.500 against itself**, on the diagonal above, and that is
arithmetic rather than luck: both games of a pair start from the same opening and are played by the
same function, so they run identically and the pair is one win and one loss. It is the invariant
the whole harness is pinned to — anything else means the pairing or the colour assignment is broken.

The summary also **flags a ladder that is not clean**. `minimax:4` beats depth 3 but not depth 2,
so it reports both rather than quoting the higher number. Connect 4 has genuine odd/even depth
effects and the position benchmark agrees they are real, so that is worth seeing rather than
smoothing.

#### Two things about the openings, both measured

Every player here is deterministic, so two games from the same position are the same game move for
move. All the variety comes from starting a few random plies in — which makes how those openings
are chosen the whole design.

**They must be distinct.** `ai/match.py` draws them at random *with replacement*, which at its
defaults gives **30 distinct openings for 50 pairs** of Connect 4 — one appearing four times. Forty
per cent of a "100-game" match is the same games replayed, while the standard error still divides
by 100 and understates itself by about 1.3×. The ladder draws from `ai/oracle.py`'s `openings_at`
instead, which enumerates every distinct position at a ply, and **raises rather than repeating**.

**They must be level.** Most openings are already won for somebody — 920 of Connect 4's 1,120 at
four plies — and a pair from a decided opening is forced to 0.5 as soon as both players convert it.
Those pairs go quiet exactly when the players get good enough to be worth telling apart. Starting
only from drawn positions was measured over the same 50 pairs:

| | all openings | drawn only |
|---|---|---|
| `minimax:2` vs `minimax:6` | 0.425 ± 0.045 *(noise)* | **0.390 ± 0.042 *(significant)*** |
| `minimax:4` vs `minimax:6` | 0.390 ± 0.044 | **0.315 ± 0.041** |

A comparison that could not be told from noise becomes one that can. `minimax:2` against
`minimax:4` stays level either way — those two really are close, and the position benchmark says so
independently. Connect 4 looks its openings up in the corpus, which is how it knows the value of a
position four plies in that its own solver cannot reach; tic-tac-toe just solves them.

The control that says all this works is a **perfect** player on tic-tac-toe:

```
$ python3 zero.py ladder --player minimax:9

  opponent                         score  record             verdict
  random                  0.935 +/- 0.017  +87 =13 -0         beats
  minimax:1               0.810 +/- 0.024  +62 =38 -0         beats
  minimax:2               0.510 +/- 0.007  +2 =98 -0          level
  minimax:9               0.500 +/- 0.000  +0 =100 -0         level
```

**Zero losses on every rung**, which is what perfect play must produce and is the strongest single
check on the harness. Under the earlier unbalanced ladder the same player "lost" 21 games to
`minimax:1` — not blunders, just lost openings it had been handed. It also never *beats* the strong
rungs, because from a level tic-tac-toe position nobody can: 100 draws out of 100 against itself.

### Teaching a network Connect 4, and knowing early whether it is working

A Connect 4 run is measured in hours where a tic-tac-toe one is measured in seconds, so the point
is not to train a network and see — it is to **know it is working long before it finishes**. The
corpus and the ladder exist for this. So does everything below.

Two trunks, chosen by the board rather than copied from a paper. Tic-tac-toe keeps its dense
layers: a 3x3 kernel on a 3x3 board is a dense layer wearing weight-sharing constraints, and the
board is not translation invariant anyway. Connect 4 gets a **residual tower, 64 filters and five
blocks**, because its threats genuinely are local shapes — three of mine with a gap means the same
thing in every column, which is exactly what a shared filter encodes.

#### The search had to be turned inside out

Batch-1 inference is not *a* cost, it is very nearly the whole cost: a Connect 4 forward pass is
1101µs alone and 111µs amortised in a batch of sixty-four, and MCTS asks once per simulation.

What cannot be batched is one tree's simulations — they are sequential by construction, each going
where the previous ones' statistics send it. Taking several leaves from a single tree needs virtual
loss and **changes what the tree explores**. So the batching happens somewhere else entirely: the
search became a **generator** that yields the position it needs and is sent the answer, and
self-play advances sixty-four *separate* games in lockstep, evaluating their pending positions
together. Every game runs an ordinary sequential search and gets the tree it would have had alone.

| | per generation | | |
|---|---|---|---|
| | one game at a time | batched | |
| Tic-tac-toe, 80 games | 3.43s | **1.16s** | 3.0x |
| Connect 4, 64 games | 75.8s | **15.2s** | 5.0x |

That it is only a speed change is asserted rather than hoped: `tests/zero/test_selfplay.py` plays
the same twelve games at batch 1, 2, 5, 12 and 64 and requires **identical examples**, and the
tic-tac-toe training trace was bit-identical across the generator refactor for forty generations.

Grading is batched too, and there it is free — 22,100 independent positions with no tree involved,
which took 37s one at a time and takes **3.8s** in chunks.

#### What a run leaves behind

Every generation appends one JSON line to a metrics file, flushed as it goes, because the run whose
history is most worth having is the one that died at hour three. `plot.py` turns it into a
self-contained page of charts — standard library only, so torch stays the single dependency:

```bash
python3 zero.py --game connect4 train --generations 30 --metrics runs/connect4.jsonl
python3 plot.py runs/connect4.jsonl --open      # works mid-run, too
```

The fields are chosen so a flat curve can be *diagnosed*, not just observed. `target_entropy` is
the one worth knowing about: policy loss cannot fall below the entropy of the targets it is
fitting, so a loss that has flattened *at* that value is a network fitting its targets perfectly
and the fault is in the search producing them. Tic-tac-toe's `c_puct` being too low looked exactly
like that.

#### Surviving the machine going away

A run measured in hours needs to be restartable, so a checkpoint carries its optimiser state — Adam
keeps a running mean and variance per parameter, and a resume that dropped them would take its
first steps back unmomented. It is written to a temporary file and renamed into place, because
whatever interrupts a long run is just as likely to interrupt the write meant to survive it.

```bash
python3 zero.py --game connect4 train --games 2000 --generations 30 \
    --out models/connect4-best.pt --latest models/connect4-latest.pt \
    --metrics runs/connect4-long.jsonl --resume --commit-every 2
```

The same command line launches and relaunches: `--resume` continues from `--latest` when that file
exists and starts at generation one when it does not. `--commit-every` pushes `--latest` and the
metrics to the branch as the run produces them, so a lost machine costs a couple of generations
rather than the run.

Two files rather than one, and the distinction matters. `--out` holds the *best* network by the
oracle benchmark, which is what you want to play against; `--latest` holds the most recent one,
which is what a resume must read. Resuming from the best checkpoint replays every generation since
the last improvement — the metrics file grows a repeated generation number where the seam is, which
is how this was caught.

#### The first run

Thirty generations, 64 games each at 50 simulations — **10.7 minutes**:

| Generation | 1 | 6 | 12 | 18 | 24 | 30 |
|---|---|---|---|---|---|---|
| Agreement with perfect play | 60.6% | 69.7% | 70.8% | 71.0% | 72.4% | **74.6%** |
| Value head error | 1.21 | 0.94 | 0.91 | 1.02 | 0.94 | **0.87** |
| Policy loss | 1.928 | 1.853 | 1.821 | 1.796 | 1.750 | **1.728** |
| Target entropy | 1.848 | 1.678 | 1.657 | 1.629 | 1.611 | 1.597 |

It starts above `random` (54.1% on the same 22,100 positions) and climbs steadily toward
`minimax:4` (79.1%). Policy loss stays clear of the target entropy, so the network still has room
rather than having perfectly fitted bad targets. Time splits 14.3s self-play, 3.8s grading, 3.3s
learning per generation.

The checkpoint after those thirty generations, graded on all three tiers:

| | opening | random play | real play |
|---|---|---|---|
| Agreement | 74.6% | 78.0% | 73.9% |
| As first / second | 75.9% / 69.4% | 78.6% / 77.5% | 79.2% / 68.6% |

**It is not a strong player yet** — thirty generations is a de-risking run, not a training run. What
it establishes is that the curve rises, the instruments read it, and a generation costs 21s, so
reaching `minimax:4` is minutes away rather than a day.

### What alpha-beta and move ordering are worth

Leaves reached from the empty Connect 4 board, all three searching the same tree:

| Depth | Plain negamax | Alpha-beta, centre-first | Alpha-beta, left-to-right |
|---|---|---|---|
| 4 | 2,401 | 97 | 735 |
| 5 | 16,807 | 391 | 3,350 |
| 6 | 117,649 | 685 | 13,160 |
| 7 | 810,504 | 3,128 | 53,290 |

All three pick the same move — alpha-beta only skips branches that provably cannot change the
answer, and `tests/connect4/test_search_equivalence.py` checks that score for score, not just
best move to best move. The last two columns are the same search differing only in which column
`legal_moves` offers first, which is why move ordering lives in the game rather than in `ai/`.

### What the evaluation is worth

`ai/match.py` plays two move-choosers off over hundreds of games, pairing every opening so both
sides get it. Connect 4's evaluation was built one term at a time and measured each time — and
measuring it against a **fixed opponent** rather than against its own previous version reversed
most of the conclusions. Depth 4, 300 games, all against the same search evaluating every
position as zero:

| Evaluation | Score |
|---|---|
| Open threes only *(shipped)* | **0.700 ± 0.019** |
| Open threes + a centre-column bonus | 0.663 ± 0.020 |
| + open twos, playability, direction weighting | 0.388 ± 0.025 |
| A centre-column bonus alone | 0.425 ± 0.013 |

Every one of those terms won its head-to-head against the version immediately before it, and
the endpoint is much worse than the start. A chain of pairwise wins is not a chain of
improvements. The shipped row re-scores 0.649 ± 0.018 on a seed it was not tuned against and
0.705 ± 0.029 at depth 5, so it is not an artefact of the depth it was tuned at. Against a
random mover it wins 200 games out of 200.

The other half of it is that **an evaluation returning zero is not passive**. With nothing to
choose between moves the search takes the first one generated, and `legal_moves` generates the
centre column first, so zero *is* the policy "take the middle unless there's a tactic" — a
strong Connect 4 heuristic. A term with an opinion about every quiet position overrides that
everywhere, including where it has nothing useful to say. Saying nothing is worth more than any
positional term tried on top of it.

## Playing against other engines

`games/chess/run_engine.py` plays Mildred against a third-party UCI engine, using the async
client in `games/chess/uci/`. It is the only external yardstick here — everything else the
project measures, it measures against itself.

The engines are not in the repository. Drop the binaries under `third-party-engines/`, which is
gitignored, and name one with `--engine`, either by key or by path:

| Key | Path | Reputed |
|---|---|---|
| `stockfish` | `stockfish/Mac/stockfish-11-64` | 3495 ELO |
| `saruman` | `saruman/engine/Saruman` | 1457 ELO |
| `feeks` | `feeks/feeks.sh` | 970 ELO |
| `pos` | `pos/pos.sh` | 111 ELO |

```bash
python3 -m games.chess.run_engine                     # one game against Stockfish
python3 -m games.chess.run_engine -e saruman -n 20    # twenty games, alternating colours
python3 -m games.chess.run_engine --depth 5 --skill 3 # deeper, against a stronger setting
```

It reports as a match rather than a game, reusing `MatchResult` from `ai/match.py`, because one
game against an engine says almost nothing — the result is dominated by colour and opening, and
the standard error makes that obvious rather than leaving it to be forgotten. Colours alternate
each game for the same reason.

## Development

Nothing to install for the engine or its tests: both are pure standard library. The one optional
extra is PyTorch, for the learned player in `ai/zero/` — the tests that need it skip themselves
when it is absent, so `tests.test_all` passes on a clean checkout either way.

Run the suites individually while iterating:

```bash
python3 -m unittest tests.chess.test_moves -v               # rules and legality
python3 -m unittest tests.chess.test_undo -v                # full-game replay, forwards and back
python3 -m unittest tests.chess.test_permutations -v        # perft
python3 -m unittest tests.chess.test_conformance -v         # chess against the shared game contract
python3 -m unittest tests.chess.test_search_equivalence -v  # search scores, position by position

python3 -m unittest tests.connect4.test_wins -v             # every line of four, in every direction
python3 -m unittest tests.connect4.test_board -v            # the sentinel invariants, fuzzed
python3 -m unittest tests.connect4.test_conformance -v      # connect 4 against the same contract
python3 -m unittest tests.connect4.test_evaluation -v       # symmetry, bounds and threat masks
python3 -m unittest tests.connect4.test_solver -v           # the exact solver against 280 pinned answers
python3 -m unittest tests.connect4.test_corpus -v           # the solved corpus, checked against itself
python3 -m unittest tests.connect4.test_ladder -v           # the ladder, and its self-play identity

python3 -m unittest tests.tictactoe.test_perfect_play -v    # the engine against an independent solver
python3 -m unittest tests.tictactoe.test_permutations -v    # perft, and the published game census
python3 -m unittest tests.tictactoe.test_conformance -v     # tic-tac-toe against the same contract
python3 -m unittest tests.tictactoe.test_board -v           # the eight lines, re-derived independently
python3 -m unittest tests.tictactoe.test_encoding -v        # planes, actions and the eight symmetries

python3 -m unittest tests.test_oracle -v                    # the benchmark, calibrated on known players
python3 -m unittest tests.zero.test_mcts -v                 # PUCT against a perfect evaluator (no torch)
python3 -m unittest tests.zero.test_selfplay -v             # the training targets (no torch)
python3 -m unittest tests.zero.test_net -v                  # the network and its checkpoints (torch)
```

`tests/zero/test_mcts.py` and `tests/zero/test_selfplay.py` deliberately need no PyTorch. The
parts of a learned player most likely to be wrong — the tree and the training targets — are the
parts that can be checked without a single trained weight, and a bug in either looks exactly like
a network that has not learned yet.

`tests/chess/test_search_equivalence.py` is the net for changing the search, which has no
oracle but itself. It scores every root move over a reproducible corpus rather than only the
move that wins — two searches can agree on the best move and disagree about everything else.
Widen `CORPUS_SIZE` and `DEPTHS` there before touching the search, and expect minutes: cost
grows as branching^depth.
