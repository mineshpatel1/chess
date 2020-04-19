import os
import asyncio
import inspect

import log
from game.board import Board, Move
from ai.algorithms import random_move, alpha_beta
from ai.benchmark import simulate_game_async
from uci.protocol import start_engine

ENGINE_DIR = 'third-party-engines'
STOCKFISH = 'stockfish/Mac/stockfish-11-64'  # 3495 ELP
SARUMAN = 'saruman/engine/Saruman'  # 1457 ELO
MINNOW = 'minnow/minnow'  #
FEEKS = 'feeks/feeks.sh'  # 970 ELO
POS = 'pos/pos.sh'  # 111 ELO

ENGINE = MINNOW


async def main():
    transport, eng = await start_engine(os.path.join(ENGINE_DIR, ENGINE), limit=1)
    eng.send_line('quit')
    # await eng.set_skill(20)

    async def engine_move(board):
        await eng.set_position_from_board(board)
        return await eng.get_best_move()

    # await simulate_game_async(lambda b: alpha_beta(b, 3), engine_move)

    # for i in range(10):
    #     log.info('Simulating game...')
    #     await simulate_game_async(engine_move, lambda b: alpha_beta(b, 3), print_moves=False, print_summary=False)
    await eng.quit()


if __name__ == '__main__':
    asyncio.run(main())