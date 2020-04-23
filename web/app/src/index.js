import React from 'react';
import ReactDOM from 'react-dom';

import './index.css';
import { Button } from './controls';
import { ChessGame } from './chess';
import { Connect4Game } from './connect4';


class Game extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            game: 'connect4',
        };
    }

    setGame(game) {
        this.setState({'game': game});
    }

    render() {
        let { game } = this.state;
        return (
            <div className="container">
                <div className="panel">
                    <Button label="Chess" onClick={() => {this.setGame('chess');}}/>
                    <Button label="Connect 4" onClick={() => {this.setGame('connect4');}}/>
                </div>
                {
                    game === 'chess' &&
                    <ChessGame />
                }
                {
                    game === 'connect4' &&
                    <Connect4Game />
                }
            </div>
        );
    }
}

// ============================================================
  
ReactDOM.render(
    <Game />,
    document.getElementById('root')
);
