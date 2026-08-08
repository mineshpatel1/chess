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
| `games/tictactoe/encoding.py` | What a network sees: two planes, nine shared actions, eight symmetries |
| `ai/search.py` | Negamax with alpha-beta pruning, and a random mover |
| `ai/perft.py`, `ai/simulate.py` | Move enumeration, and playing two move-choosers off |
| `ai/match.py` | Playing them off a few hundred times, which is how an evaluation is judged |
| `ai/oracle.py` | Exact play, and grading any player against it in every position |
| `ai/players.py` | Naming a player — `minimax:9`, `model:best.pt+mcts:200` — in one place |
| `ai/zero/mcts.py` | PUCT search over a tree of paths, with no rollouts |
| `ai/zero/net.py` | The network: input, two hidden layers of 64, a policy head and a value head |
| `ai/zero/selfplay.py` | Games against itself, and the targets they produce |
| `ai/zero/train.py` | The generation loop, graded against the oracle as it goes |
| `play.py` | A terminal front end for any game in the registry |
| `zero.py` | Training, grading and comparing learned players |
| `tests/conformance.py` | The tests every game must pass |
| `tests/chess/` | Rules, replay/undo, perft, and search scores, all driven by chess positions |
| `tests/connect4/` | The same, plus the exhaustive win-detection oracle and the eval properties |
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

### Results

The committed checkpoint, trained for 250 generations of 80 self-play games at 150 simulations —
**17 minutes on four CPU threads**, and 27KB of weights.

**Can it be beaten? No — not even with the search switched off.**

| Player | vs perfect play | vs *any* opponent |
|---|---|---|
| **Raw policy, no search** | first: 8 lines, second: 107 — **unbeaten** | 620W / 126D — **0 losses** |
| + 25 simulations | first: 10, second: 130 — **unbeaten** | 625W / 140D — **0 losses** |

Every line, both seats, no losses. The network alone holds a draw against perfect play and beats
everything that misplays — which is the whole of what tic-tac-toe asks of a player.

**Does it know the game?** Less completely, and that is where the remaining gap is:

| Player | Overall | As first | As second | Wrong |
|---|---|---|---|---|
| Raw policy, no search | 97.90% | 97.32% | 98.47% | 97 |
| + 25 simulations | 99.34% | 99.17% | 99.52% | 30 |
| + 200 simulations | 99.78% | 99.63% | 99.95% | 10 |
| + 800 simulations | 99.87% | 99.75% | **100.00%** | 6 |
| `minimax:9` (perfect by construction) | 100% | 100% | 100% | 0 |

The 97 raw-policy mistakes are all at plies 3–7, and 91 of them throw away a draw rather than a
win. Every one is in a position the model never reaches when it is the one playing — which is
exactly why the table above reads "unbeaten" while this one does not read 100%.

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

**One action space, over perspective-relative planes.** Plane 0 is always the mover's marks, so
the network learns one player's problem and every game teaches it about both seats. Nine actions,
not nine per player — a split head halves the data each half sees and invites the two halves to
disagree about which block is which.

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

### Coverage, and an honest caveat

On-policy self-play is what makes AlphaZero work, and it is also why a trained network only ever
sees the positions it plays into. Measured here: self-play reached **366 of the 4,520** decision
positions, and the network scored 96.7% on those and 78.8% on everything else.

Starting a share of self-play games from a random position (`OPENING_PLIES`) took the
all-positions score from 80.3% to 96.8% — the largest single improvement measured here, larger
than any change to the network, the loss or the search. Nothing played during those plies is
recorded, so every training target still comes from a real search.

**But it is worth being suspicious of that number.** It moved the "knows the game" score a long
way and the "cannot be beaten" score not at all — the network was already unbeatable without it.
A correct AlphaZero is supposed to reach the state space through `c_puct` and root Dirichlet
noise, so needing to force random starts is at least as likely to be evidence that the exploration
is not pulling its weight as it is a genuine improvement. Random openings also bias the training
distribution toward positions real play never reaches, which is a real cost paid for a metric of
debatable value.

All three knobs — the temperature schedule, `c_puct` and the Dirichlet weight — are parameters of
`train()` so the alternative can be measured rather than argued about.

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
