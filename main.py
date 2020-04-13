import time
import chess

import log
from ai import algo
from engine.bitboard import *
from engine.constants import PIECE_POINTS

def main():
    bb = Board()
    log.info(bb)
    log.info(bitboard_to_str(bb.pawns[WHITE]))

    log.info(bb.piece_at(D2))
    log.info(bb.piece_at(A2))
    log.info(bb)


if __name__ == '__main__':
    main()
