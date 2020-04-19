import asyncio
from typing import Any, Optional, List, Tuple

from uci.protocol import BaseCommand, UciProtocol


class StockfishProtocol(UciProtocol):
    async def get_fen(self):
        """Gets FEN from the display function. Stockfish only, move into that subclass later."""

        class Command(BaseCommand):
            def start(self, engine):
                engine.send_line('d')  # Display function

            def line_received(self, engine: UciProtocol, line: str):
                if line.startswith('Fen:'):
                    _fen = ' '.join(line.split(' ')[1:])
                    self.done(_fen)

        return await self.communicate(Command)


async def start_engine(path) -> Tuple[asyncio.BaseTransport, StockfishProtocol]:
    """Starts a UCI protocol chess engine"""
    loop = asyncio.get_event_loop()
    transport, eng = await loop.subprocess_exec(StockfishProtocol, path)
    await eng.ping()  # Make sure the engine is ready
    return transport, eng
