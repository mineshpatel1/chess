import log
from board import Position


def main():
    x = 38
    file, rank = Position._index_to_rankfile(x)
    log.info((file, rank))

    i = Position._rankfile_to_index(file, rank)
    log.info(i)



if __name__ == '__main__':
    main()
