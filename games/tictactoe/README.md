# Tic-tac-toe

The smallest thing that can satisfy the game contract — three files, no bitboard module, and a move
that is just the number of a cell — and the only game here that is **solved**. That is what makes
it useful: the same alpha-beta that plays chess approximately plays this one perfectly, and any
player can be graded against the truth in every position rather than against an opponent. It is
also the game the learned player in `ai/zero/` is developed on. See the
[main README](../../README.md) for the shared search and the AlphaZero mechanics.

```
[✕][1][○]
[3][✕][5]
[6][7][○]
```

```bash
python3 play.py                                              # play it in the terminal
python3 zero.py benchmark --player minimax:9                 # grade against perfect play
python3 zero.py train                                        # train a network from scratch
```

| Module | What is in it |
|---|---|
| `constants.py` | Nine cells, eight winning lines, and the order moves are tried in |
| `board.py` | `TicTacToe`: two 9-bit integers, and a `SOLVED_DEPTH` of nine |
| `evaluation.py` | Open twos, which matter only below the depth the game is solved at |
| `encoding.py` | What a network sees: one signed plane, nine shared actions |

## Representation

Nine cells have exactly eight winning lines, so they are enumerated once at import and a win is
`marks & line == line` — no padding, no shifts, no bitboard module.

What it has instead is **`SOLVED_DEPTH = 9`**: the whole game tree fits inside a search, so
`play.py` searches all of it by default and the engine plays perfectly rather than well. A full
solve from an empty board takes about 30ms, which is why this needs no transposition table, no
iterative deepening and no memoisation — the shared search is already fast enough to be exact.

Its move ordering is derived rather than declared: cells are tried in order of how many winning
lines run through them, which puts the centre first (four lines), then the corners (three), then
the edges (two). Since every opening move is theoretically drawn, the search has nothing to choose
between them and takes the first one generated — so that ordering is also the entire opening book.

## Perft and the census

Small enough to have been counted exhaustively long ago, so unlike Connect 4 it has a genuine
external oracle, and unlike chess's the whole table fits:

| Depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Nodes | 9 | 72 | 504 | 3,024 | 15,120 | 54,720 | 148,176 | 200,448 | 127,872 |

The first five are the falling factorial `9!/(9−d)!`: a win needs three marks, so the first
player's third at ply 5 is the earliest a game can end, and nothing before it constrains anything.
Depth 6 is the first count that is not — 5,760 of the 60,480 six-move sequences are games somebody
had already won.

Stronger still is the **census**, which perft cannot give because counting sequences cannot tell a
won game from a drawn one. Every one of the 255,168 playable games is played out in the suite and
its result recorded, against the published totals:

| First player wins | Second player wins | Draws | Total | Reachable positions |
|---|---|---|---|---|
| 131,184 | 77,904 | 46,080 | **255,168** | **5,478** |

## Perfect play

The whole game is searchable, so the engine is tested against the answer rather than against a
proxy for it. `tests/tictactoe/test_perfect_play.py` writes an **independent** memoised minimax —
no alpha-beta, no move ordering, no negamax — and checks the engine against it three ways:

| Claim | How it is checked |
|---|---|
| The search plays optimally | Its move is one the solver would play, in all 5,478 positions |
| The pruning is sound | Alpha-beta's score agrees with unpruned minimax, in all 5,478 positions |
| The engine cannot be beaten | Played against *every* opponent line, as first player and as second |

All of it runs in about 1.5 seconds, and the last row is the one a person cares about: there is no
sequence of moves that beats it from either side.

## Grading against perfect play

Because the game is solved, `ai/oracle.py` can walk **every one of the 5,478 reachable positions**,
ask the player for a move, and compare it with the set of moves that hold the position's value. No
committed corpus is needed — the answers are computed on the spot.

That is a much harder test than playing matches, and a more useful one. A player only ever meets
its own blunders through an opponent willing to punish them, and losing lines are precisely the
ones a decent opponent never steers into. Walking every position has no opponent in it and no
sampling either.

```
$ python3 zero.py benchmark --player model:models/tictactoe-best.pt --value

  overall       97.9% optimal (4423/4520), 97 blunders, 0.0228 mean value lost
  as first      97.3% optimal (2358/2423), 65 blunders, 0.0289 mean value lost
  as second     98.5% optimal (2065/2097), 32 blunders, 0.0157 mean value lost
  by ply       0:100%  1:100%  2:100%  3:98%  4:95%  5:99%  6:98%  7:98%  8:100%
  value head   0.1910 mean squared error vs truth
```

The report splits by seat, and that split is the point: a player can be excellent from one seat and
hopeless from the other, and an aggregate hides it behind an average. That is not hypothetical — it
is what the 2021 attempt in this repository's history did, and it went unnoticed for months.

A move counts as wrong only if it **changes the result**. A slower win is not a mistake, and
neither is an arbitrary choice between moves of equal value: `solve()` returns 1/0/−1 with no depth
term, so a mate in three scores the same as a mate in one.

## The ladder

Being solved also makes this the **control** that says the match harness works. A perfect player
must take zero losses on every rung:

```
$ python3 zero.py ladder --player minimax:9

  opponent                         score  record             verdict
  random                  0.935 +/- 0.017  +87 =13 -0         beats
  minimax:1               0.810 +/- 0.024  +62 =38 -0         beats
  minimax:2               0.510 +/- 0.007  +2 =98 -0          level
  minimax:9               0.500 +/- 0.000  +0 =100 -0         level
```

It never *beats* the strong rungs either, because from a level tic-tac-toe position nobody can: 100
draws out of 100 against itself. Under an earlier unbalanced ladder the same player "lost" 21 games
to `minimax:1` — not blunders, just lost openings it had been handed, which is why the ladder now
starts only from drawn positions. Tic-tac-toe solves its own openings rather than looking them up.

## The learned player

The network keeps **dense layers, 9 → 64 → 64 → (9 policy, 1 value)**: a 3x3 kernel on a 3x3 board
is a dense layer wearing weight-sharing constraints, and the board is not translation invariant
anyway. One signed plane beat two binary ones by 1.2 points.

`SELECTION_METRIC` picks the best checkpoint on **agreement** here, not on the ladder — the ladder
saturates once the network plays perfectly, so selecting on it means choosing arbitrarily among
ties, while agreement is still resolving real differences.

### Reproducing the committed checkpoint

```bash
python3 zero.py train
```

That is the whole command — the CLI defaults *are* the recipe, and training is deterministic given
`--seed`. Written out in full, every argument here is also the default:

```bash
python3 zero.py train \
    --generations 400 \        # 45 min on four CPU threads
    --games 80 \               # self-play games per generation
    --simulations 50 \         # MCTS simulations per move
    --steps 60 \               # gradient steps per generation
    --seed 1 \
    --out models/tictactoe-best.pt
```

Grading it afterwards is two more commands:

```bash
python3 zero.py benchmark --player model:models/tictactoe-best.pt            # the network alone
python3 zero.py benchmark --player model:models/tictactoe-best.pt+mcts:50    # with search
```

The shared training parameters and what each was measured against are in the
[main README](../../README.md#training). The exploration values there were tuned on this game:

| Change | On-policy agreement |
|---|---|
| Starting point | 80.3% |
| Temperature over the whole game, not the first 3 plies | 93.7% |
| `c_puct` 1.5 → 5.0 | **97.5%** |
| *(random openings, for comparison)* | *96.8%* |

Forcing a share of self-play games to start from a random position looked like the single largest
improvement in the implementation. It was not an improvement — it was covering for two exploration
parameters being set wrong. `c_puct` alone, over 90 generations at 50 simulations, is an inverted U
with a clear peak:

| c_puct | 1.5 | 3.0 | **5.0** | 8.0 |
|---|---|---|---|---|
| Agreement | 92.83% | 95.38% | **95.84%** | 94.65% |

At 1.5 the visit counts were concentrating before the alternatives had been checked, so the network
was fitting targets that were confidently slightly wrong — and its policy loss sat at the entropy
floor of those targets, which is what "fitting them perfectly" looks like. **Random openings are
now off by default:** a correct AlphaZero reaches the state space through PUCT and root noise, and
needing to force random starts on top is a sign that one of those is mistuned.

### Results

The committed checkpoint (`models/tictactoe-best.pt`): **45 minutes on four CPU threads**, 25KB of
weights, purely on-policy — no random openings, no symmetry augmentation, no supervision, nothing
but self-play.

**Can it be beaten? No — not even with the search switched off.**

| Player | vs *any* opponent, as first | vs *any* opponent, as second |
|---|---|---|
| **Raw policy, no search** | 107W / 4D — **0 losses** | 432W / 125D — **0 losses** |

Every line, both seats, no losses. The network alone holds a draw against perfect play and beats
everything that misplays, which is the whole of what tic-tac-toe asks of a player.

**Does it know the game?** Less completely, and that is where the remaining gap is:

| Player | Overall | Wrong (of 4,520) |
|---|---|---|
| Raw policy, no search | 98.30% | 77 |
| + 25 simulations | 99.69% | 14 |
| + 50 simulations | 99.87% | 6 |
| + 200 simulations | **99.98%** | **1** |
| `minimax:9` (perfect by construction) | 100% | 0 |

The raw-policy mistakes are all mid-game and nearly all throw away a draw rather than a win. Every
one is in a position the model never reaches when it is the one playing — which is exactly why the
table above reads "unbeaten" while this one does not read 100%.

Tic-tac-toe's 4,520 decision positions collapse to **627 distinct boards** once the eight
symmetries are folded together, so any comparison against another implementation has to agree on
which is being counted before the numbers mean anything.

## Tests

```bash
python3 -m unittest tests.tictactoe.test_perfect_play -v    # the engine against an independent solver
python3 -m unittest tests.tictactoe.test_permutations -v    # perft, and the published game census
python3 -m unittest tests.tictactoe.test_conformance -v     # against the shared game contract
python3 -m unittest tests.tictactoe.test_board -v           # the eight lines, re-derived independently
python3 -m unittest tests.tictactoe.test_encoding -v        # planes, actions and the eight symmetries
```

Being solved is also what lets the MCTS tests in `tests/zero/` run with no trained weights at all:
hand the search a **perfect** evaluator built from this game's oracle and it must play perfectly;
hand it one that knows nothing and it must *improve* with more simulations.

| Evaluator | 10 sims | 50 sims | 200 sims | 800 sims |
|---|---|---|---|---|
| Perfect (the oracle) | 100% | 100% | 100% | 100% |
| Ignorant (flat priors, zero value) | 89.7% | 97.1% | 99.3% | 100% |
