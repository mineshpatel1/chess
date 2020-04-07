import re
import types
import logging
from typing import Any


TAB = ' ' * 4
CLIENT_NAME = 'chess'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
MSG_FORMAT = '[%(levelname)-8s][%(asctime)s] %(message)s'


class ShellColor:
    """
    Usage: print(ShellColor.RED + '...' + ShellColor.RESET)
    See the colors at https://en.wikipedia.org/wiki/ANSI_escape_code#3.2F4_bit
    """
    RESET = '\033[0m'
    UNDERLINE = '\033[4m'
    BOLD = '\033[1m'
    FRAMED = '\033[51m'

    # Basic 8
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # From extended 256 colour palette
    GREY = '\033[38;5;8m'
    DEEP_RED = '\033[38;5;196m'

    # Bright 8
    BRIGHT_BLACK = '\033[1;30m'
    BRIGHT_RED = '\033[1;31m'
    BRIGHT_GREEN = '\033[1;32m'
    BRIGHT_YELLOW = '\033[1;33m'
    BRIGHT_BLUE = '\033[1;34m'
    BRIGHT_MAGENTA = '\033[1;35m'
    BRIGHT_CYAN = '\033[1;36m'
    BRIGHT_WHITE = '\033[1;37m'

    # Background 8
    BACKGROUND_BLACK = '\033[40m'
    BACKGROUND_RED = '\033[41m'
    BACKGROUND_GREEN = '\033[42m'
    BACKGROUND_YELLOW = '\033[43m'
    BACKGROUND_BLUE = '\033[44m'
    BACKGROUND_MAGENTA = '\033[45m'
    BACKGROUND_CYAN = '\033[46m'
    BACKGROUND_WHITE = '\033[47m'

    # Background Bright 8
    BACKGROUND_BRIGHT_BLACK = '\033[100m'
    BACKGROUND_BRIGHT_RED = '\033[101m'
    BACKGROUND_BRIGHT_GREEN = '\033[102m'
    BACKGROUND_BRIGHT_YELLOW = '\033[103m'
    BACKGROUND_BRIGHT_BLUE = '\033[104m'
    BACKGROUND_BRIGHT_MAGENTA = '\033[105m'
    BACKGROUND_BRIGHT_CYAN = '\033[106m'
    BACKGROUND_BRIGHT_WHITE = '\033[107m'


class BasicFormatter(logging.Formatter):
    def __init__(self):
        logging.Formatter.__init__(self, fmt=MSG_FORMAT, datefmt=DATE_FORMAT)


class ColouredFormatter(logging.Formatter):
    def __init__(self):
        # Adjust base format string so that it adds placeholders to colourise
        # and reset the colour of the level name.
        match_str = r'%\(levelname\)-\d.*?s'
        match = re.search(match_str, MSG_FORMAT)
        repl = f'%(colour)s{match.group(0)}%(reset)s'
        _format = '%(reset)s' + re.sub(match_str, repl, MSG_FORMAT)

        logging.Formatter.__init__(
            self,
            fmt=_format,
            datefmt=DATE_FORMAT,
        )
        self.colours = {
            'DEBUG': ShellColor.GREY,
            'INFO': ShellColor.WHITE,
            'WARNING': ShellColor.YELLOW,
            'ERROR': ShellColor.RED,
            'CRITICAL': ShellColor.DEEP_RED,
        }

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in self.colours:
            record.colour = self.colours[levelname]
            record.reset = ShellColor.RESET

        return logging.Formatter.format(self, record)


def create_logger(
        name: str,
        level: int = logging.DEBUG,
        formatter: logging.Formatter = ColouredFormatter,
):
    """
    Returns a new logging.Logger instance that can be used to log messages with debug,
    info, warning, error and critical.

    Args:
        name: Name of the logger, will be included as part of the log message.
        level: Default level for outputting logs. E.g. logging.INFO will mean only logs of INFO and higher are logged.
        formatter: Specifies the format class to use for this instance of the logger.
    """

    def log_newline(self, lines=1):
        # Switch handler, output a blank line
        if hasattr(self, 'file_handler'):
            self.removeHandler(self.file_handler)
            self.addHandler(self.blank_file_handler)

        self.removeHandler(self.console_handler)
        self.addHandler(self.blank_handler)

        for _ in range(lines):
            self.info('')

        # Switch back
        if hasattr(self, 'file_handler'):
            self.removeHandler(self.blank_file_handler)
            self.addHandler(self.file_handler)

        self.removeHandler(self.blank_handler)
        self.addHandler(self.console_handler)

    logger = logging.getLogger(name)
    if len(logger.handlers) == 0:
        logger.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter())
        logger.addHandler(console_handler)
        logger.console_handler = console_handler

        blank_handler = logging.StreamHandler()
        blank_handler.setLevel(logging.DEBUG)
        blank_handler.setFormatter(logging.Formatter(fmt=''))
        logger.blank_handler = blank_handler

        logger.newline = types.MethodType(log_newline, logger)

    logger.propagate = False
    return logger


_log = create_logger(CLIENT_NAME)


def root():
    return _log


def debug(msg: Any):
    _log.debug(msg)


def info(msg: Any):
    _log.info(msg)


def warning(msg: Any):
    _log.warning(msg)


def error(msg: Any):
    _log.error(msg)


def critical(msg: Any):
    _log.critical(msg)


def newline():
    _log.newline()


def setLevel(level: int):
    _log.setLevel(level)
