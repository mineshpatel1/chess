import time
import chess

import log
from ai import algo
from engine.position import file_rank_to_index, index_to_file_rank
from engine.bitboard import *
from engine.constants import PIECE_POINTS

def main():
    bb = Board()
    log.info(bb)
    for move in bb._pseudo_legal_moves(WHITE):
        log.info(move.uci)



if __name__ == '__main__':
    main()
