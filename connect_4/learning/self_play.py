import os
import pickle
import numpy as np
from typing import Tuple

import log
from connect_4.game import Connect4
from connect_4.learning.mcts import MCTS
from connect_4.learning.model import NUM_ACTIONS


DATASET_DIR = os.path.join(os.path.dirname(__file__), 'datasets')


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

    return p_i, values


def self_play():
    g = Connect4()
    dataset = []

    # Play a game against itself, saving board states and policy values
    while not g.is_game_over:
        mcts = MCTS(g)
        move = mcts.get_best_move()
        p_i, values = get_policy(mcts)

        log.info((g.turn, move))
        dataset.append([g.copy(), p_i])

        g.make_move(move)

    value = 0
    if g.turn is True:
        value = 1
    elif g.turn is False:
        value = -1

    for datum in dataset:
        datum.append(value)

    save_as_pickle('self_play_1', dataset)
