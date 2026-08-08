"""
Runs Mildred as a UCI engine, reading commands from stdin.

    python3 -m games.chess.uci

This is the machine-facing way in, and the only one a chess GUI can use: it speaks UCI on
stdin and stdout so Arena, CuteChess, Scid or a tournament manager can drive the engine.
play.py at the top level is the other way in, and it is not a substitute - that one prompts a
person and draws a board, which is no use to a GUI.
"""

from games.chess.uci.engine import UciEngine


def main():
    UciEngine().run()


if __name__ == '__main__':
    main()
