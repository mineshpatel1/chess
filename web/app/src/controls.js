import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCircleNotch } from '@fortawesome/free-solid-svg-icons';

export class Button extends React.Component {
    render() {
        return (
            <div className="button" onClick={this.props.onClick}>
                <span>{this.props.label}</span>
            </div>
        )
    }
}

export class Loading extends React.Component {
    render() {
        return (
            <div className="loading">
                <FontAwesomeIcon icon={faCircleNotch} size="lg" spin />
            </div>
        )
    }
}

export class Info extends React.Component {
    constructor(props) {  super(props); }
    static defaultProps = {
        error: false,
        message: "",      
    }

    render () {
        let { message, error } = this.props;
        let style = error ? "error" : "info";
        return (
            <div className="panel">
                <div className={"message " + style}>
                    {message}
                </div>
            </div>
        )
    }
}

export function capitalise(s) { 
    return s[0].toUpperCase() + s.slice(1); 
}

export function post_req(data) {
    return {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    };
}
