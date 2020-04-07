

STARTING_STATE = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


class Cell:
    def __init__(self):
        self.state = None


class Board:
    def __init__(self, state=STARTING_STATE):
        self.cells = []
        for _ in range(64):
            self.cells.append(None)

