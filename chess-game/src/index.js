import React from 'react';
import ReactDOM from 'react-dom';

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faChessRook, faChessKnight, faChessBishop, faChessQueen, faChessKing, faChessPawn } from '@fortawesome/free-solid-svg-icons'

import './index.css';

function capitalise(s) { 
    return s[0].toUpperCase() + s.slice(1); 
}

function post_req(data) {
    return {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    };
}

export class _Square {
    constructor(rank, file, piece, pieceColour) {
        this.rank = rank;
        this.file = file;
        this.index = (rank * 8) + (file % 8);
        this.piece = piece;
        this.pieceColour = pieceColour;
        this.selected = false;

        let odd_rank = this.rank % 2
        if ((this.index + odd_rank) % 2 === 0) {
            this.colour = 'black';
        } else {
            this.colour = 'white';
        }
    }
}

class Button extends React.Component {
    render() {
        return (
            <div className="button" onClick={this.props.onClick}>
                <span>{this.props.label}</span>
            </div>
        )
    }
}

class Square extends React.Component {
    render() {
        let {square, mode, turn} = this.props;

        let square_classes;
        if (square.selected) {
            square_classes = `square selected`;
        } else {
            square_classes = `square ${square.colour}`;
        }

        if (
            (mode === 'selectSquare' || square.pieceColour === turn) &&
            mode != 'end'
        ) {
            square_classes += ' clickable';
        }

        let piece_classes = `piece ${square.pieceColour}`;
        let icons = {
            'p': faChessPawn,
            'r': faChessRook,
            'n': faChessKnight,
            'b': faChessBishop,
            'q': faChessQueen,
            'k': faChessKing,
        };

        
        return (
            <div className={square_classes} onClick={() => {
                if (mode === 'selectSquare') {
                    this.props.onClick();
                }
            }}>
                {
                    square.piece &&
                    <div className={piece_classes} onClick={() => {
                        if (mode === 'selectPiece') {
                            this.props.onClick();
                        }
                    }}>
                        <FontAwesomeIcon icon={icons[square.piece]} size="lg" fixedWidth />
                    </div>
                }
            </div>
        )
    }
}

class Game extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            board: null,
            turn: null,
            selected: null,
            mode: 'selectPiece',
            error: null,
            message: null,
        };
    }

    selectPiece(i) {
        if (this.state.mode === 'selectPiece') {
            this.setState({'selected': i, 'mode': 'selectSquare'});
        }
    }

    selectSquare(i) {
        if (this.state.mode === 'selectSquare' && i === this.state.selected) {
            this.setState({'selected': null, 'mode': 'selectPiece'});
        } else if (this.state.mode === 'selectSquare') {
            const requestOptions = post_req({ 'start_pos': this.state.selected, 'end_pos': i });
            fetch('/makeMove', requestOptions).then(res => res.json())
                .then(data => {
                    if (data.end) {
                        this.setState({'board': data.board, 'message': data.end, 'selected': null, 'mode': 'end', 'error': null});
                    } else if (data.error) {
                        this.setState({'error': data.error, 'selected': null, 'mode': 'selectPiece'});
                    } else if (data.board) {
                        this.setState({
                            'board': data.board,
                            'turn': data.turn,
                            'fen': data.fen,
                            'selected': null,
                            'mode': 'selectPiece',
                            'error': null,
                        });
                    }
                })
                .catch(() => {
                    this.setState({'error': "Server error: could not make move."});
                });
        }
    }

    loadGame() {
        const requestOptions = post_req({ 'state': '3Q4/5k2/p1R5/8/3PQ3/1q5p/7P/7K w - - 3 58' });
        fetch('/loadGame', requestOptions).then(res => res.json())
            .then(data => {
                this.setState({
                    'board': data.board,
                    'turn': data.turn,
                    'fen': data.fen,
                    'selected': null,
                    'error': null,
                    'message': null,
                    'mode': 'selectPiece',
                });
            })
            .catch(() => {
                this.setState({'error': "Server error: cannot load game."})
            });
    }

    newGame() {
        fetch('/newGame').then(res => res.json())
            .then(data => {
                this.setState({
                    'board': data.board,
                    'turn': data.turn,
                    'fen': data.fen,
                    'selected': null,
                    'error': null,
                    'message': null,
                    'mode': 'selectPiece',
                });
            })
            .catch(() => {
                this.setState({'error': "Server error: cannot start new game."})
            });
    }

    render() {
        let {board, mode, turn, fen, message} = this.state;
        let rows = [];
        for (let i = 0; i < 8; i++) {
            let squares = [];
            for (let j = 0; j < 8; j++) {
                if (board) {
                    let _square = new _Square(
                        Math.abs(i - 7), j, 
                        board[i][j].piece,
                        board[i][j].piece_colour
                    );
                    if (_square.index === this.state.selected) {
                        _square.selected = true;
                    }
                    squares.push(_square);  // Push empty square
                } else {
                    squares.push(new _Square(Math.abs(i - 7), j));  // Push empty square
                }
            }
            rows.push(squares);
        }
        let info = null;
        if (turn) {
            if (mode === 'end') {
                info = message;
            } else {
                info = capitalise(turn) + " to move";
            }
        }

        return (
            <div className="container">
                <div className="state">{fen}</div>
                <div className="panel">
                    <Button label="New Game" onClick={() => { this.newGame(); }}/>
                    <Button label="Load Game" onClick={() => { this.loadGame(); }}/>
                </div>
                <div className="reactive_square">
                    <div className="board">
                        {
                            rows.map((rank, i) => {
                                return (
                                    <div className="row" key={i}>
                                        {
                                            rank.map((square, j) => {
                                                return (
                                                    <Square 
                                                        square={square} key={j} mode={mode} turn={turn}
                                                        onClick={() => {
                                                            if (mode === 'selectPiece' && square.pieceColour === turn) {
                                                                this.selectPiece(square.index);
                                                            } else if (mode === 'selectSquare') {
                                                                this.selectSquare(square.index);
                                                            }
                                                        }} 
                                                    />
                                                );
                                            })
                                        }
                                    </div>
                                );
                            })
                        }
                    </div>
                </div>
                <div className="panel">
                    <div className="message info">
                        {info}
                    </div>
                </div>
                <div className="panel">
                    <div className="message error">
                        {this.state.error}
                    </div>
                </div>
            </div>
        );
    }
}
  
// ============================================================
  
ReactDOM.render(
    <Game />,
    document.getElementById('root')
);
