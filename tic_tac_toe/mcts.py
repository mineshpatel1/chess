import random
import numpy as np
from typing import List

import log
from tic_tac_toe.game import Game

WIN_SCORE = 10
NUM_ITERATIONS = 20000
DEPTH = 15
EXPLORATION_CONSTANT = 0.5
LOW_BOUND = -9999999.0
HIGH_BOUND = 9999999.0

class Node:
    def __init__(self, state: Game, parent: int = None):
        self.state = state
        self.children = []
        self.parent = parent
        self.visit_count = 0
        self.win_score = 0

    @property
    def id(self):
        return self.state.id


class MCTS:
    def __init__(
        self, board: Game, iterations: int = NUM_ITERATIONS, depth: int = DEPTH,
        exploration_constant: float = EXPLORATION_CONSTANT,
    ):
        self.iterations = iterations
        self.depth = depth
        self.exploration_constant = exploration_constant,
        self.total_n = 0
        self.player = board.turn  # Player we want a move for

        root_node = Node(board)
        self.root_node_id = root_node.id
        self.tree = {
            self.root_node_id: root_node,
        }

    def selection(self):
        leaf_node_found = False
        leaf_node_id = self.root_node_id

        while not leaf_node_found:
            node_id = leaf_node_id
            num_children = len(self.tree[node_id].children)

            if num_children == 0:
                leaf_node_found = True
            else:
                max_uct_value = LOW_BOUND
                for child_id in self.tree[node_id].children:
                    w = self.tree[child_id].win_score
                    n = self.tree[child_id].visit_count
                    N = self.total_n

                    if n == 0:
                        uct_value = HIGH_BOUND
                    else:
                        uct_value = (w / n) * np.sqrt(np.log(N) / n)  # Upper Confidence Bound applied to trees

                    if uct_value > max_uct_value:
                        max_uct_value = uct_value
                        leaf_node_id = child_id

        return leaf_node_id

    def expansion(self, leaf_node_id):
        state = self.tree[leaf_node_id].state
        child_node_id = leaf_node_id  # Default value if the game is ovre

        if not state.is_game_over:
            # Make each possible move
            children = []
            for move in state.legal_moves:
                _state = state.copy()
                _state.make_move(move)

                children.append(_state.id)
                self.tree[_state.id] = Node(_state, parent=leaf_node_id)
                self.tree[leaf_node_id].children.append(_state.id)

            # Choose random child node
            child_node_id = random.choice(children)
        return child_node_id

    def simulation(self, child_node_id):
        self.total_n += 1
        _state = self.tree[child_node_id].state.copy()

        while not _state.is_game_over:
            _state.make_random_move()  # Choose random move for simulation

        return _state.end_result

    def backpropagation(self, child_node_id, winner):
        if winner == 0:
            reward = 0
        elif winner == self.player:
            reward = 1
        else:
            reward = -1

        complete = False
        node_id = child_node_id

        while not complete:
            self.tree[node_id].visit_count += 1
            self.tree[node_id].win_score += reward

            parent_id = self.tree[node_id].parent
            if parent_id == self.root_node_id:
                self.tree[parent_id].visit_count += 1
                self.tree[parent_id].win_score += reward
                complete = True
            else:
                node_id = parent_id

    def get_best_move(self):
        # Run Monte Carlo Tree Search and build tree
        for i in range(self.iterations):
            leaf_node_id = self.selection()
            child_node_id = self.expansion(leaf_node_id)
            result = self.simulation(child_node_id)
            self.backpropagation(child_node_id, result)

        best_score = LOW_BOUND
        best_move = None
        for child_node_id in self.tree[self.root_node_id].children:
            score = self.tree[child_node_id].win_score / self.tree[child_node_id].visit_count
            if score > best_score:
                best_score = score
                best_move = self.tree[child_node_id].state.move_history[-1]
        return best_move
