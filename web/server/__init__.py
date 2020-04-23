from flask import Flask

app = Flask(__name__)

from web.server import chess
from web.server import connect_4