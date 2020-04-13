import time

import log
from engine.position import file_rank_to_index, index_to_file_rank
from engine.bitboard import *
from engine.constants import WHITE, BLACK


def main():
    start_time = time.time()


    bb = Board('rnbqkbnr/ppp1pppp/8/8/B1p1R2P/1P1Q1PB1/P1PPP1P1/RN2K1N1 b Qkq - 0 1')
    log.info(bb)

    for i in range(1):
        x = set()
        for move in bb._pseudo_legal_moves(WHITE):
            x.add(move.uci)
        # log.info(bitboard_to_str(bb._attack_bitboard(WHITE)))
        # log.info(bb.is_in_check)

    # bb = Board()
    # log.info(bb)
    # log.info(bitboard_to_str(bb._attack_bitboard(WHITE)))
    # log.info(bb.is_in_check)

    log.info(time.time() - start_time)

if __name__ == '__main__':
    main()
