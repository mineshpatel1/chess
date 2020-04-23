import log
from connect_4.game import *
from game.board import *


def main():
    fen = '3rr3/1ppb1pp1/pbnnpk1p/8/1P1P1P2/B1P2NPP/2N5/1q1QKBR1 w - - 0 38'
    g = Board(fen)

    log.info(g)

    for move in g.legal_moves:
        if move.uci == 'd1b1':
            log.info(move)


if __name__ == '__main__':
    main()

