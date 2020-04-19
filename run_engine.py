import asyncio

import log

from uci.stockfish import start_engine

STOCKFISH = "/Users/neshpatel/stockfish/Mac/stockfish-11-64"


async def main():
    moves = ['g1f3', 'd7d6', 'b1a3']

    transport, eng = await start_engine(STOCKFISH)
    await eng.set_option('Skill Level', 0)

    await eng.set_position()
    fen = await eng.get_fen()
    log.warning(fen)

    await eng.set_position(moves=moves)
    fen = await eng.get_fen()
    log.warning(fen)

    await eng.set_position(fen='rnbqkbnr/ppp1pppp/3p4/8/8/N4N2/PPPPPPPP/R1BQKB1R b KQkq - 1 2', moves=['d8d7'])
    fen = await eng.get_fen()
    log.warning(fen)

    move = await eng.get_best_move()
    log.warning(move)

    await eng.quit()

if __name__ == '__main__':
    asyncio.run(main())