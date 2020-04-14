import time

import log
from engine.position import file_rank_to_index, index_to_file_rank
from engine.game import Game
from engine.bitboard import *
from engine.constants import WHITE, BLACK

# from engine.blue_peter import *


def main():
    bb = Board()
    for m in (
        'a2a3',
        'g7g5',
        'a3a4',
        'g5g4',
        'f2f4',
        'g4f3',
    ):
        move = Move.from_uci(m)
        if move not in bb.legal_moves:
            raise Exception
        bb.make_move(move)
    log.info(bb.fen)

if __name__ == '__main__':
    main()
