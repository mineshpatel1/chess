import time
import log

from tic_tac_toe.game import *
from tic_tac_toe.mcts import MCTS


def main():
    g = Game()

    while not g.is_game_over:
        mcts = MCTS(g)
        move = mcts.get_best_move()
        g.make_move(move)
        log.info(g)


if __name__ == '__main__':
    main()

