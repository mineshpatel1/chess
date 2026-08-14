# Connect 4

The game that shows the search is not secretly chess. Two integers, a sentinel-padded board, an
exact solver, a committed corpus of solved positions, and a trained network that beats a seven-ply
search. See the [main README](../../README.md) for the shared search and the AlphaZero mechanics
this page's numbers come out of.

```
[ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][○][ ][ ][ ]
[ ][ ][ ][●][ ][ ][ ]
[ ][ ][○][●][ ][ ][ ]
[ ][●][●][○][ ][ ][ ]
 0  1  2  3  4  5  6
```

```bash
python3 play.py                                             # play it in the terminal
python3 zero.py --game connect4 benchmark --player minimax:4
python3 zero.py --game connect4 ladder --player minimax:4
```

| Module | What is in it |
|---|---|
| `constants.py` | The board's shape, and every mask derived from it |
| `bitboard.py` | The carry that drops a disc, and the shifts that find four in a row |
| `board.py` | `Connect4`: two integers, move generation, make/unmake, win detection |
| `evaluation.py` | Open threes, weighted by direction and by whether they are live |
| `encoding.py` | What a network sees: two planes of 6x7, seven shared actions |

## Representation

Two integers, one per player, over a board with a **sentinel row**: a cell is `column * STRIDE +
row` where `STRIDE` is one *more* than the number of rows, so each column carries a spare cell
above it that is never occupied. That extra bit is what makes the rest of the engine simple.

Move generation is a single addition — `(occupied + BOTTOM_ROW) & FULL_BOARD` rings a carry up
every column at once and it settles on the cell a disc would land in, with a full column's carry
absorbed by its sentinel — so there is no `height[]` array to keep in step.

Win detection is four shift chains, one per direction, halving the problem each time: `pos & (pos
>> d)` marks adjacent pairs, and the same again at `2d` marks runs of four. Neither can wrap from
the top of one column into the bottom of the next, because the sentinel is in the way.

`legal_moves` offers the **centre column first**, which is worth a factor of nineteen at depth 6.
That ordering lives in the game rather than in `ai/` precisely because it is this game's knowledge.

## What the evaluation is worth

The evaluation was built one term at a time and measured each time — against a **fixed opponent**
rather than against its own previous version, which reversed most of the conclusions. Depth 4, 300
games, all against the same search evaluating every position as zero:

| Evaluation | Score |
|---|---|
| Open threes only *(shipped)* | **0.700 ± 0.019** |
| Open threes + a centre-column bonus | 0.663 ± 0.020 |
| + open twos, playability, direction weighting | 0.388 ± 0.025 |
| A centre-column bonus alone | 0.425 ± 0.013 |

Every one of those terms won its head-to-head against the version immediately before it, and the
endpoint is much worse than the start: **a chain of pairwise wins is not a chain of improvements.**
The shipped row re-scores 0.649 ± 0.018 on a seed it was not tuned against and 0.705 ± 0.029 at
depth 5, so it is not an artefact of the depth it was tuned at.

The other half of it is that **an evaluation returning zero is not passive**. With nothing to
choose between moves the search takes the first one generated, and `legal_moves` generates the
centre column first — so zero *is* the policy "take the middle unless there's a tactic", a strong
Connect 4 heuristic. A term with an opinion about every quiet position overrides that everywhere,
including where it has nothing useful to say.

## What alpha-beta and move ordering are worth

Leaves reached from the empty board, all three searching the same tree:

| Depth | Plain negamax | Alpha-beta, centre-first | Alpha-beta, left-to-right |
|---|---|---|---|
| 4 | 2,401 | 97 | 735 |
| 5 | 16,807 | 391 | 3,350 |
| 6 | 117,649 | 685 | 13,160 |
| 7 | 810,504 | 3,128 | 53,290 |

All three pick the same move — alpha-beta only skips branches that provably cannot change the
answer, and `tests/connect4/test_search_equivalence.py` checks that score for score, not just best
move to best move. The last two columns are the same search differing only in which column
`legal_moves` offers first.

## Perft

There is no published table, so the counts are *derived* rather than looked up. A column takes six
discs and the first player's fourth disc arrives at ply 7, so nothing can fill or win inside six
plies:

| Depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Nodes | 7 | 49 | 343 | 2,401 | 16,807 | 117,649 | **823,536** | **5,673,234** |

The first six are exactly 7ᵈ. Depth 7 is `7⁷ − 7`, the seven prefixes that stack a single column
and then have one move fewer. Depth 8 is where wins start truncating lines and hand derivation
gives out; it runs at ~640,000 nodes/sec and is left out of the suite at ~9s.

## The exact solver

Grading a player here means sampling positions and solving each one exactly, so how early in a game
the corpus can reach is decided by nothing but how fast the solver is.

It was full-width negamax with a string-keyed memo and no pruning. It is now alpha-beta over a
transposition table carrying bound flags, an immediate-win shortcut, table-driven move ordering and
mirror-symmetry sharing. Time to solve one position from a **cold** table, median over sampled
positions at each ply:

| Discs on the board | 24 | 22 | 20 | 18 | 16 | 14 |
|---|---|---|---|---|---|---|
| Before | 0.29s | ~1.3s | *did not finish* | *did not finish* | *did not finish* | *did not finish* |
| After | 0.000s | 0.010s | 0.088s | 0.054s | 1.85s | 22.8s |
| Worst seen, after | 0.34s | 0.42s | 1.54s | 3.42s | 21.1s | 69.9s |

**The frontier moved eight plies**, from 24 to 16. Positions that share a table are far cheaper,
which is why a corpus is affordable much deeper than interactive use is.

Three optional hooks on `GameState` carry it generically, each defaulting to something correct and
slow so the other games are untouched:

| Hook | What Connect 4 supplies |
|---|---|
| `solver_key` | `discs[turn] + occupied` — the whole position as one exact integer, not a hash |
| `canonical_key` | the same, or its left-right mirror, whichever is smaller |
| `winning_moves` | `completions(mine) & drops(occupied)` — can I win right now, in bit operations |

The mirror is the part that could go wrong in silence, so the design removes the opportunity.
**Value bounds are stored under `canonical_key`** — sound, because a reflected board is worth
exactly what the original is. **Move-ordering hints are stored under `solver_key`**, in a separate
dictionary, and are never mirrored: the best move in a reflected position is the *reflected* move.
No move can come out of a mirrored entry because there is no mirrored entry to read one from.

`winning_moves` earns its own hook because the profiler put about half the running time in that one
question, asked at every node. Playing all seven columns and testing each board costs 18.8µs;
`completions` marks every cell that would finish a line of four and intersects it with where a disc
would land, for 4.5µs.

```bash
python3 bench.py corpus                       # re-solve the 280 pinned positions, timed: 3.9s
python3 bench.py frontier --plies 20 18 16    # what fresh positions cost now
python3 bench.py verify                       # re-solve the corpus ourselves, as deep as we reach
```

That the answers did not change is checked rather than claimed. `tests/connect4/solved.py` pins 280
positions — value *and* full optimal-move set, forty at each even ply from 22 to 34 — generated by
the unoptimised solver before any of this landed. Beside it sit **mirror invariance**
(`solve(p) == solve(mirror(p))`, which needs no oracle at all and is exactly the fault symmetry
sharing could introduce) and the **published solution**: Connect 4 was solved in 1988 by Allis and
independently by Allen, so facts about it exist outside this repository.

## The solved corpus

Tic-tac-toe can be graded by solving every position on the spot. Connect 4 has ~4.5 × 10¹², so the
answers are computed once, ahead of time, and committed: **`ai/corpora/connect4.txt`, 33,300
positions, every legal move in each of them valued exactly.**

Every move, not just the best ones — `benchmark` scores a player by how much value its move gave
away, so a file holding only the optimal set could not tell a thrown-away draw from a thrown-away
win.

Three tiers, by how the positions were chosen, and **they are never averaged together**:

| Tier | Plies | How chosen | Count |
|---|---|---|---|
| `E` | 0–6 | **every** distinct position — enumerated, not sampled | 22,100 |
| `R` | 7–34 | seeded random play, 200 per ply | 5,600 |
| `P` | 7–34 | games between alpha-beta players, deviating 15% of the time | 5,600 |

The opening is enumerated because it fits — 7, 49, 238, 1,120, 4,263 and 16,422 distinct positions
per ply, so plies 0–6 are four times the whole tic-tac-toe state space and still trivial. There is
no sampling to be biased, and it stops at six because ply seven is the earliest a game can end. It
is also the tier that matters most: Connect 4 is a first-player win, and the opening is where that
win is kept or thrown away.

`R` and `P` stay apart because they measure different things. Random play reaches positions no
sensible game visits; play between decent players never asks a player to recover from a bad
position, which is where blunders live. Every ply is sampled, odd and even, because ply parity *is*
whose turn it is.

```
$ python3 zero.py --game connect4 benchmark --player minimax:4
```

| | opening (22,100) | random play (5,600) | real play (5,600) |
|---|---|---|---|
| `random` | 54.1% | 59.2% | 72.1% |
| `minimax:4` | **79.1%** | **95.5%** | **91.5%** |

Two things there justify the whole design. The opening is *much* harder than anywhere else — 79.1%
against 95.5% on deep random positions — which is exactly the region a sampled-only corpus would
have covered worst. And the tiers disagree about which is harder depending on who is asked: `P` is
easier than `R` for the random player and harder for `minimax:4`. Averaging them would produce a
number that moved for reasons having nothing to do with the player.

That inversion is why `Report` also splits by the value of the position:

| `minimax:4` on | winning positions | drawn positions | losing positions |
|---|---|---|---|
| opening | 76.1% | **58.6%** | 100% |
| random play | 95.1% | **78.1%** | 100% |
| real play | 86.5% | **88.8%** | 100% |

**Losing positions are free marks** — if every move loses, every move is optimal — and real play
reaches far more of them (1,877 of 5,600) than random play does (1,406). Drawn positions are the
opposite: usually exactly one move holds and the rest lose, so that column is the one that
discriminates.

### Where the answers came from

Our solver reaches ply 16 and the corpus needs ply 0, which no amount of optimisation closes. The
values come from [Pascal Pons' Connect 4 solver](https://github.com/PascalPons/connect4) with its
32MB opening book. It is AGPL and is **not vendored** — it is fetched and built into the gitignored
`third-party-engines/`, and only the numbers are committed:

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

Only the **sign** of its score is kept. Pons scores by distance to the end of the game, which would
have to be decoded correctly to be used and would be silently wrong if it were not — and the sign
is all the {-1, 0, 1} convention here needs. It also matches how the benchmark grades: a slower win
is not a mistake.

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

One subtlety worth recording. Connect 4 is mirror-symmetric, so a column-orientation mismatch
between the two solvers would leave every *value* right and every *optimal move set* reversed. A
consistent relabelling turns out to be harmless — the mirror cancels — but an *inconsistent* one is
fatal, and the check has teeth: mapping the input and not the output breaks 219 of the 280 pinned
positions, exactly the ones whose optimal set is asymmetric.

## The ladder

```
$ python3 zero.py --game connect4 ladder --player minimax:4
```

| Challenger | `random` | `minimax:1` | `:2` | `:3` | `:4` | `:5` | `:6` | Highest beaten |
|---|---|---|---|---|---|---|---|---|
| `random` | 0.460 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **none** |
| `minimax:2` | 1.000 | 0.925 | 0.500 | 0.535 | 0.510 | 0.275 | 0.390 | **`minimax:1`** |
| `minimax:4` | 1.000 | 0.935 | 0.490 | 0.640 | 0.500 | 0.405 | 0.315 | **`minimax:3`** |

100 games per rung, about two minutes for the lot. `minimax:4` beats depth 3 but not depth 2, so
the summary reports both rather than quoting the higher number — Connect 4 has genuine odd/even
depth effects and the position benchmark agrees they are real.

Openings are drawn from `ai/oracle.py`'s `openings_at` and restricted to **drawn** positions,
because 920 of the 1,120 four-ply openings are already won for somebody and a pair from a decided
opening is forced to 0.5 as soon as both players convert it. Over the same 50 pairs that turned
`minimax:4` vs `minimax:6` from 0.390 ± 0.044 into 0.315 ± 0.041 — noise into a result. Connect 4
looks its openings up in the corpus, which is how it knows the value of a position four plies in
that its own solver cannot reach.

## The learned player

The network gets a **residual tower, 64 filters and five blocks**, chosen by the board rather than
copied from a paper: Connect 4's threats genuinely are local shapes — three of mine with a gap
means the same thing in every column — which is exactly what a shared filter encodes.

`SELECTION_METRIC` picks the best checkpoint on the **ladder** here, not on agreement. See
[Grading a player](../../README.md#grading-a-player) for why the choice is per game.

The run that produced a player: 20 generations × 400 games at **600 simulations** and c_puct 2.0 —
8,000 self-play games, about nine hours. Playing with only 100 simulations, 100 paired games per
rung:

| opponent | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Score | 0.965 | 0.745 | 0.835 | 0.680 | 0.705 | 0.630 | 0.635 | **0.545** |
| Losses | 3% | 20% | 14% | 27% | 23% | 31% | 35% | **39%** |
| Verdict | beats | beats | beats | beats | beats | beats | beats | **level** |

**It beats every rung the ladder has**, and plays between a seven- and an eight-ply exhaustive
search. Draws rise to 13 from 3 at the crossover, which is what closely matched players do.

The odd rungs being easier than the even one below them is a property of the *opponent*. A depth-N
search evaluates after N plies, so at odd depths the last ply is its own move: it scores a position
where it has just created threats and the reply has not been made, counting the threat and not the
refutation. There is no quiescence search to extend past that. The effect is clear at 2→3 (+0.090)
and inside noise by 4→5 (+0.025) and 6→7 (+0.005).

**Search at play time is worth a lot, and every number above understates the network.** The same
weights against `minimax:5`:

| simulations | record | score | losses |
|---|---|---|---|
| 100 | +64 =13 −23 | 0.705 | 23% |
| **600** | +76 =8 −16 | **0.800** | **16%** |

About +70 Elo for thinking longer, with no retraining. For scale, the reference configuration runs
15 iterations of 5,000 games and evaluates against depth 5; this reaches that at 8,000 games — 1.6
of their iterations — with a sixth of the search budget at play time.

Committed checkpoints: `models/connect4-latest.pt` and `models/connect4-g2000-latest.pt`, with
their metrics in `runs/`.

### The hyperparameters were not measurable at this budget

`SELF_PLAY_EXPLORATION` is 5.0, tuned on tic-tac-toe, and there is an argument it should be lower
here: PUCT is `Q + c · P · √N_parent / (1 + n)`, so 50 simulations over seven columns gives each
child about seven visits and the exploration term swamps Q throughout. The visit distributions a
30-generation run produced sat at 82% of the entropy of a uniform distribution — a search that has
concluded almost nothing.

A grid of twelve-generation runs, one seed each, was meant to settle it:

| c_puct | sims | target entropy | agreement | value MSE | game plies |
|---|---|---|---|---|---|
| 1.5 | 50 | 64% of uniform | 69.0% | 0.759 | 18.6 |
| 2.5 | 50 | 74% | 70.7% | 0.812 | 17.4 |
| 5.0 | 50 | 87% | 69.9% | 0.704 | 17.1 |
| 1.5 | 200 | 55% | 69.9% | 0.789 | 20.2 |
| 2.5 | 200 | 63% | 69.1% | 0.864 | 20.3 |
| 2.0 | 600 | 47% | 72.9% | 0.683 | 24.2 |

**It settles nothing, and that is the finding.** Four runs of the *same* configuration differing
only in seed spread 5.5 points of agreement (64.5–69.9%, sd 2.5) and 0.176 of value MSE — wider
than every cell-to-cell difference above. Twelve-generation single-seed runs cannot see effects of
the size hyperparameters plausibly have; detecting two points would need five or six seeds a cell.

What does clear that floor: thirty generations reached 74.6% where twelve averaged 68.1%. More data
is the only lever with evidence behind it, which is what the next section is about.

### Two lessons from that run

**Generation size matters more than generation count.** The run plateaued after generation 20 —
head to head, generation 30 scored 0.513 ± 0.027 against generation 20, level, while generation 20
had scored 0.642 ± 0.027 against generation 16. That looked like the network saturating at 471,000
parameters against the reference's 1.6M. It was not. Holding the game budget fixed:

| from generation 30 | self-play games | schedule | result, 300 games |
|---|---|---|---|
| generations 21–30 | 4,000 | **10 × 400** | 0.513 — level |
| generations 31–32 | 4,000 | **2 × 2,000** | **0.597 — +68 Elo, significant** |

Identical games, identical architecture, identical 600 simulations. Each generation trains on a
buffer and takes a fixed number of gradient steps, so a 400-game generation makes a small, noisy
update from a narrow slice of positions, and ten of those do not compose into the one update 4,000
games supports. **The network was starved per generation, not saturated.** A 2,000-game generation
is 3.5–4 hours and a checkpoint is only written at the end of one, which on a reclaimable machine
is an all-or-nothing bet — two of four such generations were lost minutes before being recorded.
**1,000 games is the compromise this repo settled on.**

**Agreement is a diagnostic, not the headline.** The previous best network scored 81.7% agreement
and lost 93 games in 100 to `minimax:4`; this one scores 78.8% and beats `minimax:6`. Half a point
separated two networks that were not remotely the same player, and the sign was backwards. Graded
on all three tiers, what actually happened is visible:

| | opening (22,100) | random play (5,600) | real play (5,600) |
|---|---|---|---|
| Previous best | 81.7% | 87.7% | 83.2% |
| **This network** | **78.8%** | **80.4%** | **91.7%** |
| `minimax:4` | 79.1% | 95.5% | 91.5% |

It got *worse* at two tiers and much better at the third — and the third is the one made of
positions that occur in games between competent players, where it now matches depth-4 alpha-beta.
Self-play visits those positions, so capacity moved there, away from exhaustively enumerated
openings it will rarely face and random positions no sensible game reaches.

### Continuing the run

The 2,000-game generation, with the gradient steps scaled to it:

```bash
python3 zero.py --game connect4 train \
    --games 2000 --steps 750 --simulations 600 --exploration 2.0 \
    --generations 30 --buffer-size 120000 \
    --benchmark-every 1 --lader-rungs minimax:7 minimax:8 --ladder-games 100 \
    --ladder-simulations 100 --ladder-every 1 \
    --out models/connect4-g2000-best.pt --latest models/connect4-g2000-latest.pt \
    --metrics runs/connect4-g2000.jsonl --resume --commit-every 1
```

**`--steps` is the half of "starved per generation" that a bigger generation does not fix by
itself.** 60 steps at a batch of 128 draw 7,680 positions — three fifths of a 400-game generation
and an eighth of a 2,000-game one, so most of what the larger schedule collects was never trained
on. 750 steps draw 96,000, about one and a half times the buffer. `--buffer-size` holds one
generation of that size; the 20,000 default holds under a third of it.

`--simulations` and `--exploration` are not the defaults — those are tic-tac-toe's — so they belong
on every relaunch, and a resume that drops them continues the run as a different player.
`--generations` is the generation to stop at rather than a count, so it has to be above the one
`--latest` holds or the run does nothing. `--commit-every 1` because a lost generation here is
3.5–4 hours.

## The Rust self-play engine

The 2,000-game generation above is 3.5–4 hours, which is what made it an all-or-nothing bet and
what "1,000 games is the compromise this repo settled on" was a retreat from. It takes **119
seconds** through [`rust/`](../../rust/README.md).

The network was never the reason. Timed on this machine — RTX 3080, Ryzen 9 5900X — a forward pass
costs 1,101µs alone, 111µs amortised in a batch of 64, and **2.8µs amortised in a batch of 4,096**:

| batch | 1 | 64 | 512 | 4,096 | 16,384 |
|---|---|---|---|---|---|
| positions/sec | 682 | 24,434 | 226,641 | **355,959** | 330,325 |

`selfplay.play_games` evaluates one position per game per pass, so its batch *is*
`--games-in-flight`, and raising that only moves the cost into the Python trees. At 600 simulations
the whole run sits between the two leftmost columns:

| | s/game | evals/sec | mean batch | in the forward pass |
|---|---|---|---|---|
| Python, 32 in flight | 2.724 | 3,560 | 16.8 | 84% |
| Python, 128 in flight | 2.103 | 4,949 | 66.0 | 71% |

So the card is being fed at about 1% of what it will take. Moving the board, the tree and the
driver to Rust lifts the cap to *every game in the generation*, and the tree stops being a cost at
all — 2,000 games of tree work, encoding and driver together come to 7.7s of the 119:

| 600 simulations, fresh weights | s/game | 2,000 games |
|---|---|---|
| Python engine, CPU | 2.103 | 3.5–4 h |
| Rust engine, CPU inference | 0.743 | 25 min |
| Rust engine, GPU, 400 games | 0.176 | 5.9 min |
| **Rust engine, GPU, 2,000 games** | **0.060** | **2.0 min** |

The middle row is the control that says which half did the work: the engine alone is worth about
3×, and the batch it makes possible is worth another 9× on top. A bigger generation is also a
*cheaper* one per game, because a 2,000-wide batch uses the card and a 400-wide one does not.

**The two engines play the same games — not equivalent ones, the same ones.** The search is a
literal port (f64 in the same order, children in generation order, ties broken by it, no tree
reuse, no virtual loss) and the random numbers are CPython's, Mersenne Twister through
`gammavariate` and `choices` and `random.Random('1:37')`'s SHA-512 seeding, so the root noise and
the sampled move match too. `tests/zero/test_fast.py` plays a generation with each and requires
identical planes, policy targets, value targets and game lengths; `cargo test` checks the same
tree against pinned answers, and perft 1–8 against [the counts above](#perft) — depth 8 in 0.03s
against ~9s.

It is optional in the way PyTorch is optional. Not built, and `tests.test_all` skips those tests
and a run uses the Python driver.

## Tests

```bash
python3 -m unittest tests.connect4.test_wins -v             # every line of four, in every direction
python3 -m unittest tests.connect4.test_board -v            # the sentinel invariants, fuzzed
python3 -m unittest tests.connect4.test_conformance -v      # against the shared game contract
python3 -m unittest tests.connect4.test_evaluation -v       # symmetry, bounds and threat masks
python3 -m unittest tests.connect4.test_solver -v           # the solver against 280 pinned answers
python3 -m unittest tests.connect4.test_corpus -v           # the solved corpus, checked against itself
python3 -m unittest tests.connect4.test_ladder -v           # the ladder, and its self-play identity
```
