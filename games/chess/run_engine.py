"""
Plays Mildred against a third-party UCI engine.

    python3 -m games.chess.run_engine

Run from the repository root: ENGINE_DIR is relative to the working directory, not to this
file, and the binaries it points at are gitignored.
"""

import os
import asyncio

import log
from games.chess.board import *
from ai.search import random_move, alpha_beta
from ai.simulate import simulate_game_async
from games.chess.uci.stockfish import start_engine

ENGINE_DIR = 'third-party-engines'
STOCKFISH = 'stockfish/Mac/stockfish-11-64'  # 3495 ELP
SARUMAN = 'saruman/engine/Saruman'  # 1457 ELO
FEEKS = 'feeks/feeks.sh'  # 970 ELO
POS = 'pos/pos.sh'  # 111 ELO

ENGINE = STOCKFISH


async def main():
    transport, eng = await start_engine(os.path.join(ENGINE_DIR, ENGINE))
    await eng.set_skill(0)

    async def engine_move(board):
        await eng.set_position_from_board(board)
        return await eng.get_best_move()

    board = ChessBoard()
    await simulate_game_async(board, lambda b: alpha_beta(b, 4), engine_move)
    log.info(board.pgn_uci)

    await eng.quit()


if __name__ == '__main__':
    asyncio.run(main())
