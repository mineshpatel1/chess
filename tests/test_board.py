import unittest

import position


class TestBoard(unittest.TestCase):
    def _verify_pos(self, index, file, rank, coord):
        f, r = position.index_to_rankfile(index)
        self.assertEqual((f, r), (file, rank))

        i = position.rankfile_to_index(f, r)
        self.assertEqual(i, index)

        p = position.from_index(index)
        self.assertEqual((p.file, p.rank, str(p)), (file, rank, coord))

    def test_positions(self):
        for index, file, rank, coord in [
            (0, 0, 0, 'A1'),   # Bottom-Left
            (7, 7, 0, 'H1'),   # Bottom-Right
            (56, 0, 7, 'A8'),  # Top-Left
            (63, 7, 7, 'H8'),  # Top-Right

            (12, 4, 1, 'E2'),  # Misc positions
            (38, 6, 4, 'G5'),
            (53, 5, 6, 'F7'),
        ]:
            self._verify_pos(index, file, rank, coord)


def main():
    unittest.main()


if __name__ == '__main__':
    main()