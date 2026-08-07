"""
A deterministic corpus of chess positions, used to compare two search implementations against
each other over far more ground than hand-picked test positions can cover.

Positions come from seeded random playouts, so the corpus is reproducible run to run: a
disagreement found once can be found again.
"""

import random
from typing import List

from games.chess.board import ChessBoard
from games.chess.constants import STARTING_STATE

# Playout starting points. The perft positions are here because they were chosen to be awkward
# - castling both sides, en passant that exposes a King, promotion races - so playouts from
# them wander through more of the rules than playouts from the opening do.
SEED_POSITIONS = (
    STARTING_STATE,
    'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',   # Kiwipete
    '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',                              # Position 3
    'r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1',       # Position 4
    'rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8',              # Position 5
)

# Positions at or beside a finish, where terminal scoring is what decides the move. Random
# playouts reach these only by luck, so they are pinned in by hand. These are the positions
# tests/test_search.py reasons about.
DECISIVE_POSITIONS = (
    '6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1',            # Ra8 is mate in one
    'r5k1/5ppp/8/8/8/8/5PPP/7K w - - 0 1',          # Black threatens Ra1 mate
    '7k/Q7/8/8/8/8/8/2R4K w - - 0 1',               # Rc8 mates now, Queen moves mate later
    'kn5R/3p1p2/1P1p1p2/P2p1p2/3p1P2/3p4/3P4/7K w - - 0 1',  # a5a6 stalemates, and is best
    '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1',               # A move away from stalemating Black
)


def positions(count: int = 200, seed: int = 0, max_plies: int = 60) -> List[str]:
    """
    `count` FENs of positions with at least one legal move, as a reproducible list.

    Playouts run round-robin across the seed positions rather than one seed to exhaustion, so
    a small corpus still spans all of them instead of being all opening.
    """
    rng = random.Random(seed)
    found = list(DECISIVE_POSITIONS)
    seen = set(found)

    boards = [ChessBoard(fen) for fen in SEED_POSITIONS]
    plies = 0

    while len(found) < count and plies < max_plies:
        for i, board in enumerate(boards):
            moves = list(board.legal_moves)
            if not moves:  # Playout finished, so restart it from its seed
                boards[i] = ChessBoard(SEED_POSITIONS[i])
                continue

            board.make_move(rng.choice(moves))
            fen = board.fen
            if fen not in seen and any(board.legal_moves):
                seen.add(fen)
                found.append(fen)
                if len(found) >= count:
                    break
        plies += 1

    return found[:count]
