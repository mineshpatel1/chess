import numpy as np

import log
from connect_4.game import Connect4
from connect_4.ai import alpha_beta
from connect_4.learning.mcts import MCTS
from connect_4.learning.agent import Agent, play_agents

WIN_THRESHOLD = 1.3


def self_play():
    current = Agent('current')
    best = Agent('best')

    current.self_play()  # Play self and retrain the model from this experience
    best_score, current_score, draws = play_agents(best, current)
    log.info(f"Current: {current_score}, Best: {best_score}")

    # If the current player has surpassed our previous best player, update
    if current_score > (best_score * WIN_THRESHOLD):
        log.info('Current has improved over best.')
        best.copy_model(current.model)
        best.save()


def play_classic(games: int = 10):
    best = Agent('best')
    for i in range(games):
        g = Connect4()
        while not g.is_game_over:
            if g.turn:
                move = best.best_move(g)
            else:
                move = alpha_beta(g, 5)
            g.make_move(move)
        if g.end_result == 1:
            log.info('Mildred wins!')
        elif g.end_result == -1:
            log.info('Classic wins!')
        else:
            log.info('Match Drawn.')


def main():
    current = Agent('current')
    for i in range(5):
        current.self_play()


if __name__ == '__main__':
    main()

