import unittest
import numpy as np
from src.robot_reboot.util import get_cell_at, Direction, join_quadrants


class TestUtil(unittest.TestCase):

    def test_get_cell_at_north(self):
        cell = get_cell_at(Direction.NORTH, (1, 1), 3, 3)
        self.assertEqual((0, 1), cell)

    def test_get_cell_at_south(self):
        cell = get_cell_at(Direction.SOUTH, (1, 1), 3, 3)
        self.assertEqual((2, 1), cell)

    def test_get_cell_at_west(self):
        cell = get_cell_at(Direction.WEST, (1, 1), 3, 3)
        self.assertEqual((1, 0), cell)

    def test_get_cell_at_east(self):
        cell = get_cell_at(Direction.EAST, (1, 1), 3, 3)
        self.assertEqual((1, 2), cell)

    def test_get_cell_at_when_invalid_movement(self):
        self.assertRaises(Exception, get_cell_at(Direction.EAST, (1, 1), 3, 3))

    def test_join_quadrants_when_square_quadrants(self):
        quadrant = np.arange(1, 10).reshape((3, 3))
        result = join_quadrants(quadrant, quadrant, quadrant, quadrant)
        expected = np.array([
            [1, 2, 3, 0, 3, 2, 1],
            [4, 5, 6, 0, 6, 5, 4],
            [7, 8, 9, 0, 9, 8, 7],
            [0, 0, 0, 0, 0, 0, 0],
            [7, 8, 9, 0, 9, 8, 7],
            [4, 5, 6, 0, 6, 5, 4],
            [1, 2, 3, 0, 3, 2, 1]
        ])
        np.testing.assert_equal(result, expected)
        # Quadrant is not affected
        np.testing.assert_equal(quadrant, np.arange(1, 10).reshape((3, 3)))

    def test_join_quadrants_when_not_square_quadrants(self):
        quadrant = np.arange(1, 7).reshape((3, 2))
        result = join_quadrants(quadrant, quadrant, quadrant, quadrant)
        expected = np.array([
            [1, 2, 0, 2, 1],
            [3, 4, 0, 4, 3],
            [5, 6, 0, 6, 5],
            [0, 0, 0, 0, 0],
            [5, 6, 0, 6, 5],
            [3, 4, 0, 4, 3],
            [1, 2, 0, 2, 1]
        ])
        np.testing.assert_equal(result, expected)

    def test_join_quadrants_fails_when_different_sizes_on_quadrants(self):
        quadrant = np.arange(1, 10).reshape((3, 3))
        self.assertRaises(AssertionError,
                          lambda: join_quadrants(quadrant, quadrant, quadrant, np.arange(1, 26).reshape((5, 5))))
