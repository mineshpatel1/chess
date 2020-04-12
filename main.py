import log
from engine.board import Board

def main():
    board = Board()

    by_rank = {}
    for square in board.squares:
        rank = square.pos.rank
        by_rank[rank] = by_rank.get(rank, [])

        _square = {
            'rank': square.pos.rank,
            'file': square.pos.file,
        }

        if square.piece:
            _square['piece'] = square.piece.type
            _square['piece_colour'] = square.piece.colour

        by_rank[rank].append(_square)
    log.info(by_rank)


if __name__ == '__main__':
    main()
