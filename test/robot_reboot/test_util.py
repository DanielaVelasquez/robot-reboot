import unittest
import numpy as np
from src.robot_reboot.util import get_cell_at, Direction, join_quadrants, transpose_position_to_quadrant, build_matrix, \
    generate_positions_except, generate_even_number


class TestUtil(unittest.TestCase):

    def test_get_cell_at_north(self):
        cell = get_cell_at(Direction.North, (1, 1), 3, 3)
        self.assertEqual((0, 1), cell)

    def test_get_cell_at_south(self):
        cell = get_cell_at(Direction.South, (1, 1), 3, 3)
        self.assertEqual((2, 1), cell)

    def test_get_cell_at_west(self):
        cell = get_cell_at(Direction.West, (1, 1), 3, 3)
        self.assertEqual((1, 0), cell)

    def test_get_cell_at_east(self):
        cell = get_cell_at(Direction.East, (1, 1), 3, 3)
        self.assertEqual((1, 2), cell)

    def test_get_cell_at_when_invalid_movement(self):
        self.assertRaises(Exception, get_cell_at(Direction.East, (1, 1), 3, 3))

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

    def transpose_position_to_quadrant_return_same_position_when_q1(self):
        quadrant = np.arange(1, 7).reshape((3, 2))
        pos = (2, 2)
        transposed_pos = transpose_position_to_quadrant(quadrant, pos, 1)
        self.assertEqual(pos, transposed_pos)

    def test_transpose_position_to_quadrant_when_q2_and_squared(self):
        self.assert_position_square_quadrant((0, 0), (0, 6), 2)
        self.assert_position_square_quadrant((0, 1), (0, 5), 2)
        self.assert_position_square_quadrant((0, 2), (0, 4), 2)

        self.assert_position_square_quadrant((1, 0), (1, 6), 2)
        self.assert_position_square_quadrant((1, 1), (1, 5), 2)
        self.assert_position_square_quadrant((1, 2), (1, 4), 2)

        self.assert_position_square_quadrant((2, 0), (2, 6), 2)
        self.assert_position_square_quadrant((2, 1), (2, 5), 2)
        self.assert_position_square_quadrant((2, 2), (2, 4), 2)

    def test_transpose_position_to_quadrant_when_q3_and_squared(self):
        self.assert_position_square_quadrant((0, 0), (6, 0), 3)
        self.assert_position_square_quadrant((0, 1), (6, 1), 3)
        self.assert_position_square_quadrant((0, 2), (6, 2), 3)

        self.assert_position_square_quadrant((1, 0), (5, 0), 3)
        self.assert_position_square_quadrant((1, 1), (5, 1), 3)
        self.assert_position_square_quadrant((1, 2), (5, 2), 3)

        self.assert_position_square_quadrant((2, 0), (4, 0), 3)
        self.assert_position_square_quadrant((2, 1), (4, 1), 3)
        self.assert_position_square_quadrant((2, 2), (4, 2), 3)

    def test_transpose_position_to_quadrant_when_q4_and_squared(self):
        self.assert_position_square_quadrant((0, 0), (6, 6), 4)
        self.assert_position_square_quadrant((0, 1), (6, 5), 4)
        self.assert_position_square_quadrant((0, 2), (6, 4), 4)

        self.assert_position_square_quadrant((1, 0), (5, 6), 4)
        self.assert_position_square_quadrant((1, 1), (5, 5), 4)
        self.assert_position_square_quadrant((1, 2), (5, 4), 4)

        self.assert_position_square_quadrant((2, 0), (4, 6), 4)
        self.assert_position_square_quadrant((2, 1), (4, 5), 4)
        self.assert_position_square_quadrant((2, 2), (4, 4), 4)

    def test_transpose_position_to_quadrant_when_q2_and__not_squared(self):
        self.assert_position_not_square_quadrant((0, 0), (0, 4), 2)
        self.assert_position_not_square_quadrant((0, 1), (0, 3), 2)

        self.assert_position_not_square_quadrant((1, 0), (1, 4), 2)
        self.assert_position_not_square_quadrant((1, 1), (1, 3), 2)

        self.assert_position_not_square_quadrant((2, 0), (2, 4), 2)
        self.assert_position_not_square_quadrant((2, 1), (2, 3), 2)

    def test_transpose_position_to_quadrant_when_q3_and__not_squared(self):
        self.assert_position_not_square_quadrant((0, 0), (6, 0), 3)
        self.assert_position_not_square_quadrant((0, 1), (6, 1), 3)

        self.assert_position_not_square_quadrant((1, 0), (5, 0), 3)
        self.assert_position_not_square_quadrant((1, 1), (5, 1), 3)

        self.assert_position_not_square_quadrant((2, 0), (4, 0), 3)
        self.assert_position_not_square_quadrant((2, 1), (4, 1), 3)

    def test_transpose_position_to_quadrant_when_q4_and__not_squared(self):
        self.assert_position_not_square_quadrant((0, 0), (6, 4), 4)
        self.assert_position_not_square_quadrant((0, 1), (6, 3), 4)

        self.assert_position_not_square_quadrant((1, 0), (5, 4), 4)
        self.assert_position_not_square_quadrant((1, 1), (5, 3), 4)

        self.assert_position_not_square_quadrant((2, 0), (4, 4), 4)
        self.assert_position_not_square_quadrant((2, 1), (4, 3), 4)

    def test_build_maze_set_walls_correctly_and_maze_is_square(self):
        m = build_matrix(3, [(0, 2), (2, 1)])
        expected = np.array([[0, 0, 1],
                             [0, 0, 0],
                             [0, 1, 0]])
        np.testing.assert_equal(expected, m)

    def test_generate_positions_except_generates_different_positions(self):
        positions = generate_positions_except(3, 5, (3, 4))
        self.assertEqual(3, len(positions))
        self.assertTrue((3, 4) not in positions)

    def test_generate_positions_only_on_even_cells_positions(self):
        np.random.seed(26)
        positions = generate_positions_except(3, 5, (2, 4))
        for p in positions:
            self.assertTrue(p[0] % 2 == 0, f'{p[0]} should be an even value')
            self.assertTrue(p[1] % 2 == 0, f'{p[1]} should be an even value')

    def test_generate_even_number(self):
        np.random.seed(26)
        for i in range(100):
            number = generate_even_number(5)
            self.assertEqual(number % 2, 0)
            self.assertTrue(number < 5)

    def assert_position_not_square_quadrant(self, origin, target, q):
        quadrant = np.arange(1, 7).reshape(3, 2)
        self.assertEqual(target, transpose_position_to_quadrant(quadrant, origin, q))

    def assert_position_square_quadrant(self, origin, target, q):
        quadrant = np.arange(1, 10).reshape(3, 3)
        self.assertEqual(target, transpose_position_to_quadrant(quadrant, origin, q))
