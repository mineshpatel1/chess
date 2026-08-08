"""
Plays Mildred against a third-party UCI engine.

    python3 -m games.chess.run_engine                          # one game against Stockfish
    python3 -m games.chess.run_engine -e saruman -n 20         # twenty games, alternating colours
    python3 -m games.chess.run_engine --depth 5 --skill 3      # deeper, against a stronger setting

This is the only external yardstick the project has. Everything else it measures, it measures
against itself: perft against counts derived by hand, alpha-beta against plain negamax,
evaluations against each other with ai/match.py. Playing something that is not Mildred is the
only thing that says whether any of it is any good.

So it reports like a match rather than like a game. One game against an engine says almost
nothing - the result is dominated by the colour and by whatever happened in the opening - and
`MatchResult` carries the standard error that makes that obvious rather than leaving it to be
forgotten. Colours alternate for the same reason ai/match.py pairs its openings: Mildred playing
White every time measures Mildred-as-White.

The engines are not in the repository. Drop the binaries under `third-party-engines/`, which is
gitignored, and name them with `--engine` - either one of the keys below or a path.
"""

import argparse
import asyncio
import os
import sys
from typing import Optional

import log
from ai.match import MatchResult
from ai.search import alpha_beta
from ai.simulate import simulate_game_async
from games.chess.board import ChessBoard
from games.chess.uci.stockfish import start_engine

# Resolved against the repository root rather than the working directory, so this runs from
# anywhere rather than only from the root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE_DIR = os.path.join(REPO_ROOT, 'third-party-engines')

# Engines this has been run against, and the rating each is reputed to have. Kept as a table
# because the ratings are the useful part: they say what a result is worth.
ENGINES = {
    'stockfish': 'stockfish/Mac/stockfish-11-64',  # 3495 ELO
    'saruman': 'saruman/engine/Saruman',           # 1457 ELO
    'feeks': 'feeks/feeks.sh',                     #  970 ELO
    'pos': 'pos/pos.sh',                           #  111 ELO
}


def engine_path(engine: str, engine_dir: str = ENGINE_DIR) -> str:
    """A named engine or a path to one, as a path. Named engines are relative to `engine_dir`."""
    path = ENGINES.get(engine, engine)
    if not os.path.isabs(path):
        path = os.path.join(engine_dir, path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f'no engine at {path}. Drop the binary there, or pass --engine with a path. '
            f'Known names: {", ".join(sorted(ENGINES))}'
        )
    return path


async def play_match(
    engine: str,
    games: int = 1,
    depth: int = 4,
    skill: Optional[int] = None,
    print_moves: Optional[bool] = None,
) -> MatchResult:
    """
    Plays `games` games against `engine`, and returns the tally from Mildred's point of view.

    Mildred takes White in the even-numbered games and Black in the odd ones, so an odd `games`
    hands it one more White than Black.
    """
    path = engine_path(engine)
    if print_moves is None:
        print_moves = games == 1

    transport, opponent_engine = await start_engine(path)
    wins = draws = losses = 0

    async def opponent(board: ChessBoard) -> str:
        await opponent_engine.set_position_from_board(board)
        return await opponent_engine.get_best_move()

    def mildred(board: ChessBoard):
        return alpha_beta(board, depth=depth)

    try:
        if skill is not None:
            # Stockfish's own option, and harmless elsewhere: an engine that does not know it
            # ignores the setoption line rather than failing.
            await opponent_engine.set_skill(skill)

        for game in range(games):
            mildred_is_white = game % 2 == 0
            await opponent_engine.new_game()

            board = ChessBoard()
            first, second = (mildred, opponent) if mildred_is_white else (opponent, mildred)
            outcome = await simulate_game_async(
                board, first, second, print_moves=print_moves, print_summary=False
            )

            if outcome.winner is None:
                draws += 1
            elif outcome.winner == mildred_is_white:
                wins += 1
            else:
                losses += 1

            colour = 'White' if mildred_is_white else 'Black'
            log.info(f'Game {game + 1}/{games}, Mildred as {colour}: {outcome}')
            if print_moves:
                log.info(board.pgn_uci)
    finally:
        await opponent_engine.quit()

    result = MatchResult(wins, draws, losses)
    log.info(f'vs {engine} at depth {depth}: {result}')
    return result


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        '-e', '--engine', default='stockfish',
        help=f'engine name ({", ".join(sorted(ENGINES))}) or a path to a binary',
    )
    parser.add_argument('-n', '--games', type=int, default=1, help='games to play (default 1)')
    parser.add_argument('-d', '--depth', type=int, default=4, help='Mildred\'s search depth')
    parser.add_argument(
        '--skill', type=int, default=None,
        help="the opponent's Skill Level option, 0-20. Left alone if not given",
    )
    parser.add_argument(
        '--quiet', action='store_true', help='suppress move-by-move output for a single game'
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    try:
        asyncio.run(play_match(
            engine=args.engine,
            games=args.games,
            depth=args.depth,
            skill=args.skill,
            print_moves=False if args.quiet else None,
        ))
    except FileNotFoundError as missing:
        # The expected way to get this wrong is not having the binary, which deserves the
        # message rather than a traceback with the message at the bottom of it.
        log.error(str(missing))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
