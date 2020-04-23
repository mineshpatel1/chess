import os
import asyncio

import log
from game.board import *
from ai.algorithms import random_move, alpha_beta
from ai.benchmark import simulate_game_async
from uci.stockfish import start_engine

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

    await simulate_game_async(lambda b: alpha_beta(b, 4), engine_move)

    await eng.quit()


if __name__ == '__main__':
    asyncio.run(main())
