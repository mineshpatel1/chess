import time

import log
from engine.position import file_rank_to_index, index_to_file_rank
from engine.game import Game
from engine.bitboard import *
from engine.constants import WHITE, BLACK

# from engine.blue_peter import *


def main():
    bb = Board('3q1bRk/5p2/5N1p/8/8/8/2r2PPP/6K1 b')
    log.info(bb)
    for move in bb.legal_moves:
        log.info(move)
    # log.info(bitboard_to_str(bb._attack_bitboard(WHITE, ignore=KING)))
    # log.info(any(bb.legal_moves))
    # log.info(bb.is_checkmate)

if __name__ == '__main__':
    main()
