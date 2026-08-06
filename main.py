from uci.engine import UciEngine


def main():
    """Runs Mildred as a UCI engine, reading commands from stdin."""
    UciEngine().run()


if __name__ == '__main__':
    main()
