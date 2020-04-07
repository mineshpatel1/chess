import log
from pieces import Pawn
from position import Position


def main():
    p = Pawn(False, Position(0, 6))
    log.info(p.pos)
    log.info(p.valid_moves)


if __name__ == '__main__':
    main()
