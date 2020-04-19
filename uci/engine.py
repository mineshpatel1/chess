import re
import sys
from typing import Any, Optional, List

from game.board import Board, Move
from ai.algorithms import random_move, alpha_beta


def out(line):
    print(line, file=sys.stdout)


def err(line):
    print(line, file=sys.stderr)


class Parameter:
    def __init__(
            self,
            name: str,
            default_value: Any,
            min_value: Optional[int] = None,
            max_value: Optional[int] = None
    ):
        self.name = name
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.value = default_value

    @property
    def uci_type(self):
        if isinstance(self.default_value, str):
            return 'string'
        elif isinstance(self.default_value, bool):
            return 'check'
        elif isinstance(self.default_value, int):
            return 'spin'
        else:
            raise NotImplementedError

    def set_value(self, value):
        if isinstance(self.default_value, int):
            value = int(value)
        elif isinstance(self.default_value, bool):
            value = True if value == 'true' else False

        if self.min_value is not None:
            if value < self.min_value:
                return False
        if self.max_value is not None:
            if value > self.max_value:
                return False
        self.value = value
        return True


NAME = 'Mildred'
AUTHOR = 'Nesh Patel'
PARAMS = (
    Parameter('skill', 3, min_value=0, max_value=5),
)


class UciEngine:
    """
    Runs the game as a Universal Chess Interface.
    http://wbec-ridderkerk.nl/html/UCIProtocol.html
    https://en.wikipedia.org/wiki/Universal_Chess_Interface
    """

    @staticmethod
    def isready():
        out('readyok')

    def __init__(self):
        self.board = Board()
        self.params = {}

        for param in PARAMS:
            self.params[param.name] = param

    def reset_board(self):
        self.board = Board()

    def set_fen(self, fen: str):
        self.board = Board(fen=fen)

    def play_moves(self, moves: List[str]):
        for m in moves:
            move = Move.from_uci(m)
            self.board.make_move(move)

    def get_best_move(self):
        if self.params['skill'].value == 0:
            move = random_move(self.board)
        else:
            move = alpha_beta(self.board, depth=self.params['skill'].value, print_count=False)
        out(f'bestmove {move.uci}')

    def about(self):
        output = [f'id name {NAME}', f'id name {AUTHOR}']
        for name, param in self.params.items():
            opt = f'option name {name} type {param.uci_type} default {param.default_value}'
            if param.min_value is not None:
                opt += f' min {param.min_value}'
            if param.max_value is not None:
                opt += f' max {param.max_value}'
            output.append(opt)
        output.append('uciok')
        out('\n'.join(output))

    def display(self):
        out(str(self.board) + '\n')

    def run(self):
        while True:
            cmd = input().strip()

            if cmd.lower() == 'quit':
                break
            elif cmd.lower() == 'ucinewgame':
                self.reset_board()
            elif cmd.lower() == 'isready':
                self.isready()
            elif cmd.lower() == 'uci':
                self.about()
            elif cmd.lower() == 'd':
                self.display()
            elif cmd.lower() == 'go':
                self.get_best_move()
            elif cmd.lower().startswith('setoption name'):
                match = re.search(r'setoption name (?P<name>.*?) value (?P<value>.*?)$', cmd)
                if match:
                    name = match.group('name')
                    value = match.group('value')
                    if name not in self.params:
                        err(f'No such option: {name}')
                    else:
                        if not self.params[name].set_value(value):
                            err('Could not set value')
                else:
                    err('Invalid setoption instruction')
            elif cmd.lower().startswith('position'):
                fen_flag, move_flag, start_pos = False, False, False
                fen = []
                moves = []

                for item in cmd.split(' ')[1:]:
                    if item.lower() == 'fen':
                        fen_flag = True
                        move_flag = False

                    if item.lower() == 'moves':
                        fen_flag = False
                        move_flag = True

                    if item.lower() == 'startpos':
                        start_pos = True

                    if fen_flag and item.lower() != 'fen':
                        fen.append(item)

                    if move_flag and item.lower() != 'moves':
                        moves.append(item)

                if start_pos:
                    self.reset_board()

                if fen:
                    self.set_fen(' '.join(fen))

                if moves:
                    self.play_moves(moves)
