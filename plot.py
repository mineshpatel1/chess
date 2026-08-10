"""
Turning a training run's metrics into something you can look at.

    python3 plot.py runs/connect4-fresh.jsonl
    python3 plot.py runs/connect4-fresh.jsonl --out /tmp/run.html --open

The fourth root script, joining `play.py` (play), `zero.py` (learn) and `bench.py` (measure). It
reads what `ai.zero.metrics` wrote and produces one self-contained HTML page of small multiples.

**No matplotlib, no third-party anything.** The README promises that torch is the only dependency
in this project and that promise is worth more than a plotting library: charts here are inline SVG
built from the standard library, so the page opens anywhere and the repo stays installable with
nothing. Line charts of a few hundred points do not need a framework.

**Small multiples rather than one crowded chart.** The whole reason for recording eight numbers a
generation is that a flat curve has several possible causes and they are told apart by which
*other* curves moved. Overlaying them on shared axes - a loss around 2 and a rate around 0.9 -
would make most of them unreadable.

The reference lines on the agreement chart are the point of the exercise. A number climbing is
encouraging; a number climbing past what a random player scores on the same positions, and heading
for what a depth-4 search scores on them, is progress that means something. Both were measured
against this exact grading set and live in `REFERENCES`.
"""

import argparse
import html
import os
import sys
import webbrowser
from typing import Dict, List, Optional, Sequence, Tuple

from ai.zero.metrics import read

# Scores on the same positions a run is graded against, so a curve can be read against something.
# All measured with `zero.py benchmark` on the exact grading set, and recorded in the README.
#
# The middle line for Connect 4 is the important one and was nearly left out. A network that has
# learned only "play towards the middle" - the crudest true thing about the game, and one line of
# code - already scores 73.5% on the opening tier. Drawing only `random` at 54.1% makes a climb to
# 74% look like most of the way to `minimax:4` when it is barely past the trivial policy. A floor
# that flatters the curve is worse than no floor.
REFERENCES = {
    'Connect4': [('random', 0.541), ('centre column', 0.735), ('minimax:4', 0.791)],
    'TicTacToe': [('random', 0.678), ('minimax:9 — perfect', 1.0)],
}

# Which fields to draw, in reading order: the headline first, then what explains it.
CHARTS: Sequence[Tuple[str, str, str]] = (
    # First, because it is the primary metric. Agreement follows as a diagnostic: two Connect 4
    # networks half a point apart on it scored 0.055 and 0.635 against `minimax:4`, so a page that
    # led with agreement was leading with the number that hid the difference.
    ('ladder_score', 'Ladder — mean score against the rungs', 'rate'),
    ('optimal_rate', 'Agreement with perfect play (opening only)', 'rate'),
    ('value_mse', 'Value head error against truth', 'plain'),
    ('policy_loss', 'Policy loss, against its targets’ entropy', 'plain'),
    ('value_loss', 'Value loss', 'plain'),
    ('first_rate', 'Agreement as first player', 'rate'),
    ('second_rate', 'Agreement as second player', 'rate'),
    ('distinct_positions', 'Distinct positions in the buffer', 'plain'),
    ('draw_rate', 'Self-play games drawn', 'rate'),
    ('game_length', 'Mean game length, plies', 'plain'),
    ('seconds', 'Seconds per generation', 'plain'),
    # Worth a chart of its own because its effect is on every other number's *cost* rather than on
    # any of their values. Left unflushed these accumulate under weight decay and make the whole
    # loop several times slower while the work is unchanged - a day was spent looking for that in
    # the machine before looking in the weights.
    ('denormal_weights', 'Denormal weights flushed', 'plain'),
)

WIDTH, HEIGHT = 460, 240
PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 56, 14, 16, 34


def _points(records, field) -> List[Tuple[float, float]]:
    return [
        (record['generation'], record[field])
        for record in records
        if record.get(field) is not None
    ]


def _scale(values: Sequence[float], kind: str) -> Tuple[float, float]:
    """
    The vertical range a chart covers.

    A rate is drawn on 0 to 1 always. Anything else is drawn on its own range with a little room
    above and below - autoscaling a loss to its own extremes is what makes a plateau visible,
    which is exactly the shape being looked for.
    """
    if kind == 'rate':
        return 0.0, 1.0
    low, high = min(values), max(values)
    if high == low:
        return low - 1.0, high + 1.0
    margin = (high - low) * 0.1
    return low - margin, high + margin


def _svg(records, field: str, title: str, kind: str, references) -> str:
    points = _points(records, field)
    if not points:
        return ''

    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    low, high = _scale(ys, kind)
    first, last = min(xs), max(xs)
    span = (last - first) or 1

    def px(x):
        return PAD_LEFT + (x - first) / span * (WIDTH - PAD_LEFT - PAD_RIGHT)

    def py(y):
        return HEIGHT - PAD_BOTTOM - (y - low) / (high - low) * (HEIGHT - PAD_TOP - PAD_BOTTOM)

    parts = [f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{html.escape(title)}">']
    parts.append(
        f'<rect x="{PAD_LEFT}" y="{PAD_TOP}" width="{WIDTH - PAD_LEFT - PAD_RIGHT}" '
        f'height="{HEIGHT - PAD_TOP - PAD_BOTTOM}" class="frame"/>'
    )

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = low + fraction * (high - low)
        y = py(value)
        label = f'{value:.0%}' if kind == 'rate' else f'{value:,.4g}'
        parts.append(f'<line x1="{PAD_LEFT}" y1="{y:.1f}" x2="{WIDTH - PAD_RIGHT}" y2="{y:.1f}" '
                     f'class="grid"/>')
        parts.append(f'<text x="{PAD_LEFT - 8}" y="{y + 4:.1f}" class="tick end">{label}</text>')

    for name, value in references:
        if low <= value <= high:
            y = py(value)
            parts.append(f'<line x1="{PAD_LEFT}" y1="{y:.1f}" x2="{WIDTH - PAD_RIGHT}" '
                         f'y2="{y:.1f}" class="reference"/>')
            parts.append(f'<text x="{WIDTH - PAD_RIGHT - 4}" y="{y - 5:.1f}" '
                         f'class="tick end reference-label">{html.escape(name)}</text>')

    line = ' '.join(f'{px(x):.1f},{py(y):.1f}' for x, y in points)
    parts.append(f'<polyline points="{line}" class="series"/>')

    for x in (first, last):
        parts.append(f'<text x="{px(x):.1f}" y="{HEIGHT - 12}" class="tick middle">{x}</text>')
    parts.append(f'<text x="{PAD_LEFT}" y="{HEIGHT - 12}" class="tick" '
                 f'transform="translate(0,0)"></text>')

    parts.append('</svg>')
    return ''.join(parts)


def render(records, game: Optional[str], source: str) -> str:
    """The whole page: a headline, a table of where the run got to, and the charts."""
    references = REFERENCES.get(game or '', [])
    last = records[-1]
    total = sum(record.get('seconds', 0.0) for record in records)

    summary = [
        ('Generations', f'{len(records):,}'),
        ('Agreement with perfect play', f'{last.get("optimal_rate", 0):.2%}'),
        ('Value head error', f'{last.get("value_mse", 0):.4f}'),
        ('Total time', f'{total / 60:.1f} min'),
        ('Per generation', f'{total / max(len(records), 1):.1f}s'),
    ]

    cells = ''.join(
        f'<div class="figure"><h2>{html.escape(title)}</h2>'
        f'{_svg(records, field, title, kind, references if field == "optimal_rate" else [])}</div>'
        for field, title, kind in CHARTS
        if _points(records, field)
    )
    stats = ''.join(f'<div><dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd></div>'
                    for k, v in summary)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(game or 'Training run')} — training metrics</title>
<style>
  /* Three states, not two. An explicit choice stamps data-theme on the root; the default
     "system" setting stamps nothing at all, so only prefers-color-scheme separates the
     un-stamped page. Every colour is a token defined on bare :root and only *redefined*
     under the other two, which is what stops one theme's text landing on the other's ground. */
  :root {{ --ink:#191b1f; --muted:#5f6672; --line:#d5d9e0; --grid:#ecEEF2; --series:#1f6feb;
           --reference:#b4530f; --bg:#ffffff; --panel:#f7f8fa; }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ink:#e6e8ec; --muted:#949aa6; --line:#2f343c; --grid:#23272e; --series:#6ea8fe;
      --reference:#f0913f; --bg:#0f1114; --panel:#171a1f;
    }}
  }}
  :root[data-theme="dark"] {{
    --ink:#e6e8ec; --muted:#949aa6; --line:#2f343c; --grid:#23272e; --series:#6ea8fe;
    --reference:#f0913f; --bg:#0f1114; --panel:#171a1f;
  }}
  body {{ font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
          margin:0; padding:32px; background:var(--bg); color:var(--ink); }}
  header {{ max-width:1000px; margin:0 auto 24px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .source {{ color:var(--muted); font-size:13px; font-family:ui-monospace,monospace; }}
  dl {{ display:flex; flex-wrap:wrap; gap:28px; margin:20px 0 0; }}
  dt {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  dd {{ margin:2px 0 0; font-size:20px; font-variant-numeric:tabular-nums; }}
  .grid-of-figures {{ display:grid; gap:20px; max-width:1000px; margin:0 auto;
                      grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); }}
  .figure {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
             padding:12px 14px 4px; }}
  .figure h2 {{ font-size:13px; font-weight:600; margin:0 0 4px; color:var(--muted); }}
  svg {{ width:100%; height:auto; display:block; }}
  .frame {{ fill:none; stroke:var(--line); }}
  .grid {{ stroke:var(--grid); }}
  .series {{ fill:none; stroke:var(--series); stroke-width:2; stroke-linejoin:round; }}
  .reference {{ stroke:var(--reference); stroke-dasharray:4 3; }}
  .reference-label {{ fill:var(--reference); }}
  .tick {{ font-size:10px; fill:var(--muted); font-variant-numeric:tabular-nums; }}
  .end {{ text-anchor:end; }} .middle {{ text-anchor:middle; }}
</style></head>
<body>
<header>
  <h1>{html.escape(game or 'Training run')} — training metrics</h1>
  <div class="source">{html.escape(source)}</div>
  <dl>{stats}</dl>
</header>
<main class="grid-of-figures">{cells}</main>
</body></html>
"""


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('metrics', help='the JSONL file a training run wrote')
    parser.add_argument('--out', default=None, help='where to write the page (default: alongside)')
    parser.add_argument('--game', default=None, help='which game, for the reference lines')
    parser.add_argument('--open', action='store_true', help='open it in a browser afterwards')
    args = parser.parse_args(argv)

    records = read(args.metrics)
    if not records:
        raise SystemExit(f'{args.metrics} has no complete generations in it yet')

    game = args.game or _guess_game(args.metrics)
    out = args.out or os.path.splitext(args.metrics)[0] + '.html'
    with open(out, 'w') as handle:
        handle.write(render(records, game, os.path.abspath(args.metrics)))

    print(f'{len(records)} generations -> {out}')
    if args.open:
        webbrowser.open(f'file://{os.path.abspath(out)}')


def _guess_game(path: str) -> Optional[str]:
    """From the filename, so the usual case needs no flag. Wrong guesses only cost a reference."""
    name = os.path.basename(path).lower()
    for game in REFERENCES:
        if game.lower() in name.replace('-', '').replace('_', ''):
            return game
    return None


if __name__ == '__main__':
    sys.exit(main())
