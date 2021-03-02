import unittest
from src.robot_reboot.util import get_cell_at, Direction


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
