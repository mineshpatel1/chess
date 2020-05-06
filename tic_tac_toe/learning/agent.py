import os
import pickle
import numpy as np
import multiprocessing
from typing import Tuple

import log
from tic_tac_toe.game import Game
from tic_tac_toe.learning.mcts import MCTS, NUM_ITERATIONS
from tic_tac_toe.learning.model import NUM_ACTIONS
from tic_tac_toe.learning.model import NeuralNet

SELF_PLAY_ITERATIONS = 100
TRAINING_ITERATIONS = 5
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'datasets')


def _self_play(iteration: int):
    """Play a game against itself, saving board states and policy values."""

    log.info(f'Playing game...')
    g = Game()
    sub_dataset = []
    while not g.is_game_over:
        mcts = MCTS(g, iterations=NUM_ITERATIONS, neural_net=None)
        move = mcts.get_best_move()
        p_i, values = get_policy(mcts)
        sub_dataset.append([g.copy(), p_i])
        g.make_move(move)

    end_player = g.turn
    for datum in sub_dataset:
        if end_player == datum[0].turn:
            datum.append(1)
        else:
            datum.append(-1)
    return sub_dataset


class Agent:
    def __init__(self, name: str):
        self.name = name
        self.model = NeuralNet()
        self.self_play_iterations: int = SELF_PLAY_ITERATIONS
        self.training_iterations: int = TRAINING_ITERATIONS
        self.model.load(self.model_file)

    @property
    def model_file(self):
        return os.path.join(MODEL_DIR, self.name + '.h5')

    def self_play(self, iterations: int = None):
        iterations = iterations or self.self_play_iterations

        pool = multiprocessing.Pool(4)
        jobs = []
        for i in range(iterations):
            jobs.append((i,))

        result = pool.starmap(_self_play, jobs)

        dataset = []
        for sub_dataset in result:
            dataset.extend(sub_dataset)

        # for i in range(iterations):
        #     log.info(f'Playing game {i + 1} / {iterations}...')
        #
        #     g = Game()
        #     sub_dataset = []
        #     while not g.is_game_over:
        #         mcts = MCTS(g, iterations=NUM_ITERATIONS, neural_net=self.model)
        #         move = mcts.get_best_move()
        #         p_i, values = get_policy(mcts)
        #         sub_dataset.append([g.copy(), p_i])
        #
        #         g.make_move(move)
        #
        #     end_player = g.turn
        #     for datum in sub_dataset:
        #         if end_player == datum[0].turn:
        #             datum.append(1)
        #         else:
        #             datum.append(-1)
        #
        #     dataset.extend(sub_dataset)


        log.info(f"Boards generated from self play: {len(dataset)}")
        save_as_pickle('self_play', dataset)
        self.train()

    def train(self):
        dataset = load_pickle(os.path.join(DATASET_DIR, 'self_play'))
        for i in range(self.training_iterations):
            log.newline()
            log.info(f'Training cycle {i + 1}...')
            self.model.train(
                [row[0].model_input for row in dataset],
                [row[1] for row in dataset],
                [row[2] for row in dataset],
            )
        self.save()

    def predict(self, state: Game):
        return self.model.predict_move(state)

    def best_move(self, state: Game):
        """Combines the player's neural network with a Monte Carlo Tree Search."""
        mcts = MCTS(state, neural_net=self.model)
        return mcts.get_best_move()

    def copy_model(self, other_model: NeuralNet):
        self.model.copy(other_model)

    def save(self):
        self.model.save(self.model_file)


def save_as_pickle(filename, data):
    with open(os.path.join(DATASET_DIR, filename) + '.pickle', 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    with open(os.path.join(DATASET_DIR, filename) + '.pickle', 'rb') as f:
        data = pickle.load(f)
    return data


def get_policy(mcts: MCTS, tau: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    p_i = np.zeros(NUM_ACTIONS, dtype=np.integer)
    values = np.zeros(NUM_ACTIONS, dtype=np.float32)

    for child_id in mcts.root.children:
        child = mcts.tree[child_id]
        p_i[child.move] = pow(child.visit_count, (1 / tau))
        values[child.move] = (child.win_score / child.visit_count)

    p_i = p_i / (np.sum(p_i))  # Normalise
    return p_i, values


def play_agents(player1: Agent, player2: Agent, num_games: int = 10):
    player1_score = 0
    player2_score = 0
    draws = 0

    for i in range(num_games):
        log.info(f'Playing game {i + 1} / {num_games}...')
        g = Game()
        turn_no = 0
        while not g.is_game_over:
            turn_no += 1
            if g.turn:
                move = player1.best_move(g)
            else:
                move = player2.best_move(g)
            log.info(f'    Turn No: {turn_no}')
            g.make_move(move)

        if g.end_result == 1:
            player1_score += 1
            log.info('    Player 1 wins!')
        elif g.end_result == -1:
            player2_score += 1
            log.info('    Player 2 wins!')
        elif g.end_result == 0:
            log.info('    Match drawn.')
            draws += 1

    return player1_score, player2_score, draws
