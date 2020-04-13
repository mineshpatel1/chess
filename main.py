import time
import chess

import log
from ai import algo
from engine.bitboard import *
from engine.constants import PIECE_POINTS

def main():


    # print_board(BB_H1)

    # log.info(A1)
    # for sq in SQUARES:
    #     log.info(sq)
    #
    # for bb_sq in [1 << sq for sq in SQUARES]:
    #     log.info(bb_sq)
    #
    # log.info(1 << 63)
    # log.info(2 << 1)

    # b = chess.Board()
    # start_time = time.time()
    # move = algo.negamax_fast(b, 3)
    #
    # log.info(time.time() - start_time)

    # log.info(56)
    # log.info(3)
    # log.info(15 ^ 56)

    # for sq in SQUARES:
    #     log.info((sq.name, bb_str(BB_SQUARES[sq])))

    for rank in BB_RANKS:
        log.info(bitboard_to_str(rank))
    # log.info(bitboard_to_str(BB_RANK_4))

if __name__ == '__main__':
    main()
