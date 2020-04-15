import time
import chess
import log
from ai import algo
# from engine.position import file_rank_to_index, index_to_file_rank
from engine.game import Game
from engine.bitboard import *
# from engine.constants import WHITE, BLACK

# from engine.blue_peter import *


def main():
    pass
    # start_time = time.time()
    #
    # b = Game()
    # move = algo.negamax(b, 2)
    # log.info(move)
    #
    # log.info(time.time() - start_time)  #  11.115381002426147

    # start_time = time.time()
    #
    # b = chess.Board()
    # move = algo.negamax_superfast(b, 4)
    # log.info(move)
    #
    # log.info(time.time() - start_time)  # 0.039129018783569336
    #
    # start_time = time.time()
    #
    # b = Board()
    # move = algo.negamax_bitboard(b, 4)
    # log.info(move)
    #
    # log.info(time.time() - start_time)  # 0.01094675064086914

if __name__ == '__main__':
    main()
