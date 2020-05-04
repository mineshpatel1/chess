import numpy as np

import log
from tic_tac_toe.game import Game
from tic_tac_toe.learning.agent import Agent
from tic_tac_toe.minimax import alpha_beta


def main():
    current = Agent('current')
    # current.self_play()

    # g = Game()
    # while not g.is_game_over:
    #     if g.turn:
    #         log.info(g)
    #         move = input("Play [1-9]: ")
    #         g.make_move(int(move) - 1)
    #     else:
    #         move = current.predict(g)
    #         # move = alpha_beta(g, 4)
    #         g.make_move(move)
    #         # g.make_random_move()
    # log.info(g)


    losses = 0
    draws = 0
    wins = 0
    num_games = 100
    for i in range(num_games):
        g = Game()
        while not g.is_game_over:
            if not g.turn:
                # move = alpha_beta(g, 4)
                # move = current.predict(g)
                # g.make_move(move)
                g.make_random_move()
            else:
                move = current.predict(g)
                # move = alpha_beta(g, 4)
                g.make_move(move)
                # g.make_random_move()

        log.info(f'Played {i + 1}/{num_games}...')
        if g.end_result == -1:
            losses += 1
        elif g.end_result == 0:
            draws += 1
        elif g.end_result == 1:
            wins += 1
    log.info((wins, losses, draws))


if __name__ == '__main__':
    main()

