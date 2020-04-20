from __future__ import annotations

import enum
import time
import asyncio
from typing import Any, Optional, List, Tuple

import log
from game.board import Board


class CommandState(enum.Enum):
    New = 1
    Running = 2
    Done = 3


class BaseCommand:
    """
    Base class for defining commands to send to the engine.
    Should implement start and line_received. Should call done when the command has been completed and the output
    collected.
    """
    def __init__(self):
        self.state: CommandState = CommandState.New
        self.result: asyncio.Future[Any] = asyncio.Future()

    def _start(self, engine: UciProtocol):
        self.state = CommandState.Running
        self.start(engine)

    def done(self, payload: Optional[Any]= None):
        self.state = CommandState.Done
        if not self.result.done():
            self.result.set_result(payload)

    def start(self, engine: UciProtocol):
        pass

    def line_received(self, engine: UciProtocol, line: str):
        pass


class UciProtocol(asyncio.SubprocessProtocol):
    def __init__(self, limit: int = 0.5):
        self.transport = None
        self.command: Optional[BaseCommand] = None
        self.loop = asyncio.get_event_loop()
        self.return_code: asyncio.Future[int] = asyncio.Future()
        self.limit = limit
        self.all_output = {
            1: bytearray(),  # stdout
            2: bytearray(),  # stderr
        }

    def stop_command(self):
        if self.command.state != CommandState.Done:
            self.send_line('stop')

    def connection_made(self, transport: asyncio.BaseTransport):
        self.transport = transport

    def connection_lost(self, exc: Optional[Exception]):
        code = self.transport.get_returncode()
        self.return_code.set_result(code)

    def process_exited(self):
        pass

    def pipe_data_received(self, file_descriptor: int, data: bytes):
        self.all_output[file_descriptor].extend(data)

        if file_descriptor == 1:
            for line in data.decode("utf-8").split('\n'):
                line = line.strip()
                if line:
                    # log.debug(line)
                    if line and self.command:
                        self.command.line_received(self, line)

    def send_line(self, line):
        stdin = self.transport.get_pipe_transport(0)
        stdin.write(line.encode("utf-8"))
        stdin.write(b"\n")

    async def communicate(self, command_factory):
        """Communicates a command to the engine with custom functions to capture the result."""
        command = command_factory()

        # Wait for any previously unfinished jobs
        if self.command and not self.command.result.done():
            await self.command.result

        self.command = command
        command.start(self)
        # Send a stop command after the time limit has been reached
        self.loop.call_later(self.limit, self.stop_command)
        return await self.command.result

    async def quit(self):
        """Quits the engine"""
        self.send_line('stop')
        self.send_line('quit')
        await self.return_code

    async def new_game(self):
        self.send_line('ucinewgame')
        await self.ping()
        await self.set_position()

    async def ping(self):
        """Pings to see if the system is ready to recieve a command."""
        class Command(BaseCommand):
            def start(self, engine):
                engine.send_line('isready')

            def line_received(self, engine: UciProtocol, line: str):
                if line == "readyok":
                    self.done()

        return await self.communicate(Command)

    async def set_option(self, name, value):
        """Sets a parameter of the engine."""
        self.send_line(f"setoption name {name} value {value}")
        await self.ping()

    async def set_position(self, fen: str = None, moves: List[str] = None):
        """Sets the board position in the engine according to UCI protocol."""
        command = ['position']
        if fen:
            command += ['fen', fen]
        else:
            command += ['startpos']

        if moves:
            command.append('moves')
            command += moves

        self.send_line(' '.join(command))
        await self.ping()

    async def set_position_from_board(self, board: Board):
        await self.set_position(board.fen)

    async def get_best_move(self):
        """Runs the go function and gets the best move for a given board position."""

        class Command(BaseCommand):
            def start(self, engine):
                engine.send_line('go')

            def line_received(self, engine: UciProtocol, line: str):
                if line.startswith('bestmove'):
                    _move = line.split(' ')[1]
                    self.done(_move)

        return await self.communicate(Command)


async def start_engine(
    path,
    protocol: UciProtocol = UciProtocol,
    limit: float = 0.5,
) -> Tuple[asyncio.BaseTransport, UciProtocol]:
    """Starts a UCI protocol chess engine"""
    loop = asyncio.get_event_loop()
    transport, eng = await loop.subprocess_exec(protocol, path)
    eng.limit = limit
    await eng.new_game()
    return transport, eng
