"""
The solved corpus, checked without re-solving anything.

`ai/corpora/connect4.txt` is the answer key every Connect 4 result will be measured against, and it
was computed by a program that is not in this repository - Pascal Pons' solver, fetched and built
into the gitignored `third-party-engines/`. So the corpus arrives as one program's opinion, and
none of the checks here may consult that program again. They are all things the file can be asked
about on its own, or facts from outside both projects.

**The game tree checks the corpus against itself, and it is the strongest of these by far.** The
enumerated tier holds *every* position out to six discs, so for any position shallower than that,
its children are in the file too. A position is worth what its best reply is worth, negated - so
tens of thousands of entries can be checked against tens of thousands of others with no solver
involved at all. A systematic error in the generation pipeline, a column mapped the wrong way, a
sign dropped, a tier mislabelled: none of them survive that.

**The published solution checks the shallow end.** Connect 4 was solved in 1988 by Victor Allis
and independently by James Allen: the first player wins, and only by taking the centre column.
Enumerating from the empty board means those seven positions are literally the first seven lines
of the tier, so this is a direct assertion rather than a spot check.

**Our own solver checks the deep end**, but not here - it costs seconds a position, which belongs
in `bench.py verify` rather than in a suite that has to stay quick. What is here instead is the
mirror: Connect 4 reflected in its central column is the same game, so wherever the corpus holds
both a position and its reflection - which the enumerated tier always does, being closed under the
tree - their values must be reflections too.

Plies 7 to 34 are covered by the last two only. That gap is real and is stated rather than papered
over: it is bracketed by exact checks at both ends, run through the same pipeline the enumerated
tier proves out, and re-solved independently by `bench.py verify` wherever our own solver reaches.
"""

import unittest
from collections import Counter
from typing import Dict, List

from ai.corpus import CORPORA, Entry, TIERS, format_entry, load, parse, positions, values
from games.base import has_moves
from games.connect4.board import Connect4
from games.connect4.constants import COLS

CORPUS = CORPORA['Connect4']

# The last ply the corpus enumerates in full. Below this every child of an entry is also an entry.
ENUMERATED_TO = 6

# The 1988 result, as every reference states it: the centre column wins, the two beside it draw,
# and the outer four lose. Written as values to the player to move, which is the first player.
PUBLISHED_OPENING = [-1, -1, 0, 1, 0, -1, -1]


def _load_once() -> List[Entry]:
    """Loaded once for the module. 33,300 entries is a second of parsing, but not per test."""
    global _ENTRIES
    try:
        return _ENTRIES
    except NameError:
        _ENTRIES = load(CORPUS)
        return _ENTRIES


def _by_key(entries: List[Entry]) -> Dict[int, Entry]:
    return {Connect4(entry.moves).solver_key: entry for entry in entries}


class TestGameTreeConsistency(unittest.TestCase):
    """
    The corpus against itself, using the one relation every solved game satisfies.

    A position is worth the negation of the best its opponent can do afterwards. The enumerated
    tier is closed under the game tree below its last ply, so both halves of that statement are in
    the file - which turns "do we trust the external solver" into arithmetic.
    """

    @classmethod
    def setUpClass(cls):
        cls.entries = _load_once()
        cls.opening = [entry for entry in cls.entries if entry.tier == 'E']
        cls.by_key = _by_key(cls.opening)

    def test_every_move_is_worth_the_negation_of_what_it_leads_to(self):
        checked = 0
        for entry in self.opening:
            if entry.ply >= ENUMERATED_TO:
                continue  # Its children are one ply too deep to be in the file

            state = Connect4(entry.moves)
            for move, value in entry.values.items():
                state.make_move(move)
                child = self.by_key.get(state.solver_key)
                state.unmake_move()

                self.assertIsNotNone(child, f'{entry.moves} + {move} is missing from the corpus')
                self.assertEqual(
                    value, -child.value,
                    f'{entry.moves}: move {move} is worth {value}, but leads to {child.value}',
                )
                checked += 1

        # If this ever drops, the enumeration stopped being exhaustive and the check above went
        # quietly vacuous rather than failing.
        self.assertGreater(checked, 39000, 'far fewer parent/child pairs than the tier holds')

    def test_a_position_and_its_reflection_agree(self):
        """
        Free, and it is the one thing that would catch a column mapping applied in one direction
        only. Values are mirror-invariant, so a reflected entry must hold the reflected values.
        """
        checked = 0
        for entry in self.opening:
            mirrored = self.by_key.get(Connect4([COLS - 1 - m for m in entry.moves]).solver_key)
            if mirrored is None:
                continue
            self.assertEqual(
                {COLS - 1 - column: value for column, value in entry.values.items()},
                mirrored.values,
                f'{entry.moves} and its reflection disagree',
            )
            checked += 1

        self.assertEqual(len(self.opening), checked, 'the opening tier is not closed under mirror')


class TestPublishedSolution(unittest.TestCase):
    """The only claim here that neither this repository nor the solver that built the corpus made."""

    @classmethod
    def setUpClass(cls):
        cls.by_key = _by_key(_load_once())

    def test_the_seven_opening_moves_have_the_published_outcomes(self):
        empty = self.by_key[Connect4().solver_key]
        self.assertEqual(PUBLISHED_OPENING, [empty.values[column] for column in range(COLS)])

    def test_the_centre_column_is_the_only_winning_first_move(self):
        empty = self.by_key[Connect4().solver_key]
        self.assertEqual([COLS // 2], empty.optimal)
        self.assertEqual(1, empty.value, 'Connect 4 is a first-player win')


class TestTheFileIsWellFormed(unittest.TestCase):
    """
    Guards on the corpus rather than on the game, and every one of them has failed a draft.

    A benchmark quietly graded over the wrong set of positions still prints a number, so the shape
    of the set is worth asserting as loudly as the values in it.
    """

    @classmethod
    def setUpClass(cls):
        cls.entries = _load_once()

    def test_every_tier_is_present_and_named(self):
        counts = Counter(entry.tier for entry in self.entries)
        self.assertEqual({tier for tier, _ in TIERS}, set(counts))
        for tier, _ in TIERS:
            self.assertGreater(counts[tier], 1000, f'tier {tier} is too small to mean anything')

    def test_the_opening_tier_is_exhaustive(self):
        """The distinct-position counts per ply, which are a property of the game, not a choice."""
        expected = {0: 1, 1: 7, 2: 49, 3: 238, 4: 1120, 5: 4263, 6: 16422}
        counted = Counter(entry.ply for entry in self.entries if entry.tier == 'E')
        self.assertEqual(expected, dict(counted))

    def test_the_sampled_tiers_cover_every_ply_and_both_seats(self):
        """
        Both parities, which is the whole reason plies are sampled one at a time rather than every
        other one. Ply parity *is* whose turn it is, so a corpus of even plies only would ask the
        first player everything and the second player nothing - and `Report.by_seat` exists because
        a player can be strong from one seat and hopeless from the other.
        """
        for tier in ('R', 'P'):
            plies = {entry.ply for entry in self.entries if entry.tier == tier}
            self.assertEqual(set(range(7, 35)), plies, f'tier {tier} does not cover every ply')
            self.assertTrue(any(ply % 2 == 0 for ply in plies))
            self.assertTrue(any(ply % 2 == 1 for ply in plies))

    def test_no_position_is_already_over(self):
        """A finished game has no move to grade, so one in here would be graded as nothing."""
        for entry in self.entries:
            state = Connect4(entry.moves)
            self.assertIsNone(state.outcome, f'{entry.moves} is already won')
            self.assertTrue(has_moves(state.legal_moves), f'{entry.moves} is full')

    def test_the_values_cover_exactly_the_legal_moves(self):
        """
        The check that a column mapping cannot survive. If the corpus scored the wrong columns, a
        full column would carry a value and a playable one would not.
        """
        for entry in self.entries:
            self.assertEqual(
                sorted(Connect4(entry.moves).legal_moves), sorted(entry.values),
                f'{entry.moves} values columns that are not the legal ones',
            )

    def test_every_value_is_a_result_rather_than_a_distance(self):
        for entry in self.entries:
            for column, value in entry.values.items():
                self.assertIn(value, (-1, 0, 1), f'{entry.moves} column {column}')

    def test_no_position_appears_twice(self):
        keys = [Connect4(entry.moves).solver_key for entry in self.entries]
        self.assertEqual(len(keys), len(set(keys)), 'the same position is graded more than once')


class TestTheLoader(unittest.TestCase):
    """`ai/corpus.py` on its own, on lines written here rather than read from the file."""

    def test_a_line_round_trips(self):
        line = 'E 3435 -1 0 1 1 0 0 x'
        self.assertEqual(line, format_entry(parse(line), COLS))

    def test_the_empty_board_is_written_and_read_back(self):
        entry = parse('E - -1 -1 0 1 0 -1 -1')
        self.assertEqual([], entry.moves)
        self.assertEqual('E - -1 -1 0 1 0 -1 -1', format_entry(entry, COLS))

    def test_an_illegal_column_carries_no_value(self):
        entry = parse('R 0000001 1 1 1 1 1 1 x')
        self.assertNotIn(6, entry.values)
        self.assertEqual(6, len(entry.values))

    def test_the_value_and_optimal_moves_are_derived(self):
        entry = parse('E 3435 -1 0 1 1 0 0 x')
        self.assertEqual(1, entry.value)
        self.assertEqual([2, 3], entry.optimal)

    def test_a_malformed_line_is_refused_rather_than_guessed_at(self):
        for line in ('E', 'E 34', 'E 34 x x x x x x x'):
            with self.assertRaises(ValueError, msg=line):
                parse(line)

    def test_comments_and_blank_lines_are_skipped(self):
        entries = load(CORPUS)
        self.assertTrue(entries, 'the header comment swallowed the file')

    def test_positions_and_values_line_up_with_what_benchmark_wants(self):
        entries = load(CORPUS, tiers=('E',))[:50]
        lookup = values(entries)
        for state, ply in positions(entries, Connect4):
            self.assertEqual(ply, len(state.columns_played))
            self.assertEqual(sorted(state.legal_moves), sorted(lookup(state)))

    def test_an_unsolved_position_is_refused_rather_than_skipped(self):
        """A benchmark that silently dropped what it could not value would report a made-up rate."""
        lookup = values(load(CORPUS, tiers=('E',))[:10])
        with self.assertRaises(KeyError):
            lookup(Connect4([0, 1, 2, 3, 4, 5, 6, 0, 1, 2]))


if __name__ == '__main__':
    unittest.main()
