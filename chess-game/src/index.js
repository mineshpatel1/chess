import React from 'react';
import ReactDOM from 'react-dom';

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faChessRook, faChessKnight, faChessBishop, faChessQueen, faChessKing, faChessPawn } from '@fortawesome/free-solid-svg-icons'

import './index.css';

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
        let square = this.props.square;
        let mode = this.props.mode;

        let square_classes;
        if (square.selected) {
            square_classes = `square selected`;
        } else {
            square_classes = `square ${square.colour}`;
        }

        if (mode === 'selectSquare') {
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
            selected: null,
            mode: 'selectPiece',
        };
    }

    selectPiece(i) {
        if (this.state.mode === 'selectPiece') {
            this.setState({'selected': i, 'mode': 'selectSquare'});
        }
    }

    selectSquare(i) {
        if (this.state.mode === 'selectSquare') {
            const requestOptions = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 'start_pos': this.state.selected, 'end_pos': i }),
            };
            fetch('/makeMove', requestOptions).then(res => res.json()).then(data => {
                this.setState({'board': data.board, 'selected': null, 'mode': 'selectPiece'});
            });
        }
    }

    newGame() {
        fetch('/newBoard').then(res => res.json()).then(data => {
            this.setState({'board': data.board});
        });
    }

    render() {
        let board = this.state.board;
        let mode = this.state.mode;
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

        return (
            <div className="container">
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
                                                        square={square} key={j} mode={mode}
                                                        onClick={() => {
                                                            if (mode === 'selectPiece') {
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
                <div className="controls" onClick={() => { this.newGame(); }}>
                    <Button label="New Game"/>
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
