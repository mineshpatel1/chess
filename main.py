import time
import chess
import log
from engine.position import file_rank_to_index, index_to_file_rank
from engine.game import Game
from engine.bitboard import *
from engine.constants import WHITE, BLACK

# from engine.blue_peter import *


def main():
    bb = Board('8/3k4/8/8/8/8/3K4/8 w - - 0 1')
    log.info(bb)
    log.info(bb.has_insufficient_material)

if __name__ == '__main__':
    main()
