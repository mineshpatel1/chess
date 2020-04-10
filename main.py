import log
import board
import position

from pieces import Pawn
from position import Position


def main():
    b = board.Board()
    # for p in sorted(b.pieces):
    #     log.info(p)

    # for sq in b.squares2:
    #     log.info(sq)

    # for rank, squares in b._squares_by_rank2.items():
    #     log.warning(rank)
    #     for sq in squares:
    #         log.info(sq)

    log.info(b.fen)


if __name__ == '__main__':
    main()
