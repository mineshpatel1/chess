# import time
# import chess
#
# import log
# from ai import algo
# from engine.position import file_rank_to_index, index_to_file_rank
from engine.bitboard import *
# from engine.constants import PIECE_POINTS

# from engine.blue_peter import *

def main():

    bb = Board('rnbqkbnr/ppp1pppp/8/8/2p1R2P/BP1Q1PB1/P1PPP1P1/RN2K1N1 w Qkq - 0 1')
    log.info(bb)
    for move in bb._pseudo_legal_moves_from_square(Square.from_coord('E2'), WHITE):
        log.info(move.uci)




if __name__ == '__main__':
    main()
