import React from 'react';

export class Button extends React.Component {
    render() {
        return (
            <div className="button" onClick={this.props.onClick}>
                <span>{this.props.label}</span>
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
