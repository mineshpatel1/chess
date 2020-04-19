import asyncio
import time
from datetime import datetime
import chess

import log
from game.board import *
from uci.engine import UciEngine
# from ai.benchmark import simulate_game, traverse_moves
# from ai.algorithms import random_move, negamax, minimax, alpha_beta


def main():
    engine = UciEngine()
    engine.run()


if __name__ == '__main__':
    main()
