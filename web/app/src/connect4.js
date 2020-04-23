import React from 'react';
import { Button, Info, Loading, post_req, capitalise } from './controls';

export class Connect4Game extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            game: null,
            player: true,
            mode: 'end',
            message: null,
            error: null,
            loading: false,
            twoPlayer: false,
        };
    }

    togglePlayer() {
        this.setState({'player': !this.state.player})
    }

    processBoard(data, humanMove) {
        if (data.end) {
            this.setState({
                'game': data.game,
                'turn': data.turn,
                'mhn': data.mhn,
                'message': data.end,
                'mode': 'end',
                'loading': false,
            });
        } else if (data.error) {
            this.setState({'error': data.error, 'mode': 'chooseSlot'});
        } else if (data.game) {
            this.setState({
                'game': data.game,
                'turn': data.turn,
                'mhn': data.mhn,
                'error': null,
                'message': null,
                'mode': 'chooseSlot',
                'loading': false,
            }, () => {
                if (humanMove && !this.state.twoPlayer) {
                    this.aiMove();
                }
            });
        }
    }

    makeMove(file) {
        const requestOptions = post_req({ 'move': file });
        fetch('/connect4/makeMove', requestOptions).then(res => res.json())
            .then(data => {
                this.processBoard(data, true);
            })
            .catch(() => {
                this.setState({'error': "Server error: could not play move."})
            });
    }

    aiMove() {
        this.setState({
            'loading': true,
        }, () => {
            fetch('/connect4/makeMoveAi').then(res => res.json())
                .then(data => {
                    this.processBoard(data, false)
                });
        })
    }

    newGame() {
        const requestOptions = post_req({ 'player': this.state.player });
        fetch('/connect4/newGame', requestOptions).then(res => res.json())
            .then(data => {
                this.setState({
                    'game': data.game,
                    'turn': data.turn,
                    'mhn': data.mhn,
                    'error': null,
                    'message': null,
                    'mode': 'chooseSlot',
                    'loading': false,
                });
            })
            .catch(() => {
                this.setState({'error': "Server error: cannot start new game."})
            });
    }

    render() {
        let { game, turn, mode, message } = this.state;
        let player = this.state.player ? 'Red' : 'Yellow';

        let inputRow = [];
        for (let i = 0; i < 7; i++) {
            inputRow.push(1);
        }

        let rows = [];
        for (let i = 0; i < 6; i++) {
            let slots = [];
            for (let j = 0; j < 7; j++) {
                slots.push(j);
            }
            rows.push(slots)
        }

        let info;
        if (turn) {
            if (mode === 'end') {
                info = message;
            } else {
                info = capitalise(turn) + " to play";
            }
        }

        return (
            <div>
                <div className="panel">
                    <Button label="New Game" onClick={() => { this.newGame(); }}/>
                    <Button label="Load Game" onClick={() => { this.loadGame(); }}/>
                    <Button label={"Player: " + player} onClick={() => { this.togglePlayer(); }}/>
                </div>
                <div className="reactive_square">
                    <div className="board">
                        <div className="row c4 selector">
                            {inputRow.map((_, i) => {
                                if (mode == 'chooseSlot') {
                                    return (
                                        <div className="c4 square" key={i}>
                                            <div
                                                className="circle clickable" 
                                                onClick={() => { this.makeMove(i); }}
                                            />
                                        </div>
                                    )
                                }
                            })}
                        </div>
                        {rows.map((rank, i) => {
                            return (
                                <div key={i} className="row c4">
                                    {rank.map((_, j) => {
                                        let colour;
                                        if (game) {
                                            colour =  game[i][j].piece;
                                        }

                                        let className = 'circle';
                                        if (colour) {
                                            className += ' ' + colour;
                                        }

                                        return (
                                            <div className="c4 square" key={j}>
                                                <div className={className} />
                                            </div>
                                        )
                                    })}
                                </div>
                            )
                        })}
                    </div>
                </div>
                <Info message={info}/>
                <Info message={this.state.error} error={true}/>
                {
                    this.state.loading &&
                    <Loading />
                }
            </div>
        )
    }
}