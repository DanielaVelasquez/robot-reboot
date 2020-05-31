import unittest
import numpy as np
from src.models.maze import Maze
from src.models.robotreboot import RobotReboot


class TestRobotReboot(unittest.TestCase):

    def test_move_robot_north(self):
        maze_cells = np.array([
            [0],
            [0],
            [0],
            [0],
            [0]
        ])
        maze = Maze(maze_cells)
        robots = {
            "A": (4, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "N")
        self.assertEqual((0, 0), rr.robots["A"])

    def test_move_robot_north_when_wall_at_west_and_east(self):
        maze_cells = np.array([
            [0],
            [Maze.E],
            [Maze.W],
            [0],
            [0]
        ])
        maze = Maze(maze_cells)
        robots = {
            "A": (4, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "N")
        self.assertEqual((0, 0), rr.robots["A"])

    def test_move_robot_north_with_wall_at_north(self):
        maze_cells = np.array([
            [0],
            [0],
            [Maze.N],
            [0],
            [0]
        ])
        maze = Maze(maze_cells)
        robots = {
            "A": (4, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "N")
        self.assertEqual((2, 0), rr.robots["A"])

    def test_move_robot_north_with_wall_at_south(self):
        maze_cells = np.array([
            [0],
            [0],
            [Maze.S],
            [0],
            [0]
        ])
        maze = Maze(maze_cells)
        robots = {
            "A": (4, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "N")
        self.assertEqual((3, 0), rr.robots["A"])

    def test_move_robot_north_when_robot_at_border(self):
        maze_cells = np.array([
            [0],
            [0],
            [0],
            [0],
            [0]
        ])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "N")
        self.assertEqual((0, 0), rr.robots["A"])

    def test_move_robot_north_when_another_robot_on_the_way(self):
        maze_cells = np.array([
            [0],
            [0],
            [0],
            [0],
            [0]
        ])
        maze = Maze(maze_cells)
        robots = {
            "A": (4, 0),
            "B": (2, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "N")
        self.assertEqual((3, 0), rr.robots["A"])

    def test_move_robot_south(self):
        maze_cells = np.array([
            [0],
            [0],
            [0],
            [0],
            [0]
        ])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "S")
        self.assertEqual((4, 0), rr.robots["A"])

    def test_move_robot_south_when_wall_at_west_and_east(self):
        maze_cells = np.array([
            [0],
            [Maze.E],
            [Maze.W],
            [0],
            [0]
        ])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "S")
        self.assertEqual((4, 0), rr.robots["A"])

    def test_move_robot_south_with_wall_at_north(self):
        maze_cells = np.array([
            [0],
            [0],
            [Maze.N],
            [0],
            [0]
        ])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "S")
        self.assertEqual((1, 0), rr.robots["A"])

    def test_move_robot_south_with_wall_at_south(self):
        maze_cells = np.array([
            [0],
            [0],
            [Maze.S],
            [0],
            [0]
        ])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "S")
        self.assertEqual((2, 0), rr.robots["A"])

    def test_move_robot_south_when_robot_at_border(self):
        maze_cells = np.array([
            [0],
            [0],
            [0],
            [0],
            [0]
        ])
        maze = Maze(maze_cells)
        robots = {
            "A": (4, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "S")
        self.assertEqual((4, 0), rr.robots["A"])

    def test_move_robot_south_when_another_robot_on_the_way(self):
        maze_cells = np.array([
            [0],
            [0],
            [0],
            [0],
            [0]
        ])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 0),
            "B": (2, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "S")
        self.assertEqual((1, 0), rr.robots["A"])

    def test_move_robot_east(self):
        maze_cells = np.array([[0, 0, 0, 0, 0]])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "E")
        self.assertEqual((0, 4), rr.robots["A"])

    def test_move_robot_east_when_wall_at_north_and_south(self):
        maze_cells = np.array([[0, 0, Maze.N, Maze.S, 0]])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "E")
        self.assertEqual((0, 4), rr.robots["A"])

    def test_move_robot_east_with_wall_at_east(self):
        maze_cells = np.array([[0, 0, Maze.E, 0, 0]])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "E")
        self.assertEqual((0, 2), rr.robots["A"])

    def test_move_robot_east_with_wall_at_west(self):
        maze_cells = np.array([[0, 0, Maze.W, 0, 0]])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "E")
        self.assertEqual((0, 1), rr.robots["A"])

    def test_move_robot_east_when_robot_at_border(self):
        maze_cells = np.array([[0, 0, 0, 0, 0]])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 4)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "E")
        self.assertEqual((0, 4), rr.robots["A"])

    def test_move_robot_east_when_another_robot_on_the_way(self):
        maze_cells = np.array([[0, 0, 0, 0, 0]])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 0),
            "B": (0, 2)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "E")
        self.assertEqual((0, 1), rr.robots["A"])

    def test_move_robot_west(self):
        maze_cells = np.array([[0, 0, 0, 0, 0]])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 4)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "W")
        self.assertEqual((0, 0), rr.robots["A"])

    def test_move_robot_west_when_wall_at_north_and_south(self):
        maze_cells = np.array([[0, 0, Maze.N, Maze.S, 0]])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 4)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "W")
        self.assertEqual((0, 0), rr.robots["A"])

    def test_move_robot_west_with_wall_at_west(self):
        maze_cells = np.array([[0, 0, Maze.W, 0, 0]])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 4)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "W")
        self.assertEqual((0, 3), rr.robots["A"])

    def test_move_robot_west_with_wall_at_east(self):
        maze_cells = np.array([[0, 0, Maze.E, 0, 0]])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 4)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "W")
        self.assertEqual((0, 2), rr.robots["A"])

    def test_move_robot_west_when_robot_at_border(self):
        maze_cells = np.array([[0, 0, 0, 0, 0]])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 0)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "W")
        self.assertEqual((0, 0), rr.robots["A"])

    def test_move_robot_west_when_another_robot_on_the_way(self):
        maze_cells = np.array([[0, 0, 0, 0, 0]])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 4),
            "B": (0, 2)
        }
        rr = RobotReboot(maze, robots)
        rr.move_robot("A", "W")
        self.assertEqual((0, 3), rr.robots["A"])

    def test_is_a_robot_on_position_on_maze(self):
        maze_cells = np.array([[0, 0, 0, 0, 0]])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 2),
        }
        rr = RobotReboot(maze, robots)
        self.assertTrue(rr.is_a_robot_on((0, 2)))
        self.assertFalse(rr.is_a_robot_on((0, 0)))

    def test_observation(self):
        maze_cells = np.array([
            [Maze.EMPTY, Maze.S, Maze.EMPTY, Maze.E, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.N, Maze.N, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.S, Maze.EMPTY, Maze.EMPTY],
            [Maze.E, Maze.E, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ])
        maze = Maze(maze_cells)
        robots = {
            "A": (0, 2),
            "B": (0, 2),
            "C": (4, 2)
        }
        rr = RobotReboot(maze, robots)

        obs = rr.observation
        rows, cols, layers = obs.shape

        self.assertEqual(obs.shape, (5, 5, 4))
        np.testing.assert_equal(obs[:, :, 0], maze_cells)
        self.assertEqual(obs[0, 2, 1], 1)
        self.assertEqual(obs[0, 2, 2], 1)
        self.assertEqual(obs[4, 2, 3], 1)

        for i in range(rows):
            for j in range(cols):
                for layer in range(1, layers):
                    if not rr.is_a_robot_on((i, j)):
                        self.assertEqual(obs[i, j, layer], 0)


if __name__ == '__main__':
    unittest.main()
