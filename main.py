import log
from engine.position import file_rank_to_index, index_to_file_rank
from engine.bitboard import *
# from engine.constants import PIECE_POINTS


def main():
    pass
    bb = Board('rnbqkbnr/ppp1pppp/8/8/1Bp1R2P/1P1Q1PB1/P1PPP1P1/RN2K1N1 w Qkq - 0 1')

    log.info(bb)
    x = set()
    for move in bb._pseudo_legal_moves(WHITE):
        x.add(move.uci)
    log.info(x)

if __name__ == '__main__':
    main()
