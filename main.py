import chess

import log

from ai.benchmark import traverse_moves
from engine.board import *


def main():
    b = Board()
    traverse_moves(b, 4)

    b = chess.Board()
    traverse_moves(b, 4)

if __name__ == '__main__':
    main()
