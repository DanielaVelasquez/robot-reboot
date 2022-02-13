import unittest

from src.robot_reboot.classic_robot_reboot_hash import ClassicRobotRebootZobristHash


class TestClassicRobotRebootZobristHash(unittest.TestCase):
    def test_robots_count(self):
        zobrist_hash = ClassicRobotRebootZobristHash()
        self.assertEqual(4, zobrist_hash.robots_count)

    def test_maze_shape(self):
        zobrist_hash = ClassicRobotRebootZobristHash()
        self.assertEqual((31, 31), zobrist_hash.maze_shape)

    def test_empty(self):
        zobrist_hash = ClassicRobotRebootZobristHash()
        self.assertEqual(0, zobrist_hash.empty)

    def test_get_value_for_valid_entries(self):
        zobrist_hash = ClassicRobotRebootZobristHash()
        for r in range(0, 31, 2):
            for c in range(0, 31, 2):
                for robot in range(4):
                    value = zobrist_hash.get_value((r, c), robot)
                    if value is None:
                        self.fail("Not value found for robot "+str(robot) + " at position "+ str((r, c)))

    def test_get_value_for_invalid_entries(self):
        zobrist_hash = ClassicRobotRebootZobristHash()
        for r in range(1, 31, 2):
            for c in range(1, 31, 2):
                for robot in range(4):
                    value = zobrist_hash.get_value((r, c), robot)
                    if value is not None:
                        self.fail("Value found for robot "+str(robot) + " at invalid position "+ str((r, c)))

