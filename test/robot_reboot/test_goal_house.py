import unittest
from src.robot_reboot.goal_house import RobotRebootGoalHouse
from src.robot_reboot.goals_house_exceptions import InvalidRobotIdException


class TestGoalHouse(unittest.TestCase):
    def test_init(self):
        goal = RobotRebootGoalHouse(0, (0, 2))
        self.assertEqual(0, goal.robot_id)
        self.assertEqual((0, 2), goal.house)

    def test_init_fails_when_robot_id_below_0(self):
        self.assertRaises(InvalidRobotIdException, lambda: RobotRebootGoalHouse(-1, (0, 2)))
