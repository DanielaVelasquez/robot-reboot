import numpy as np

from src.exceptions.robot_reboot.factory import UnsupportedMazeSize
from src.robot_reboot.game import RobotRebootGame
from src.robot_reboot.game_configuration import RobotRebootGameConfiguration
from src.robot_reboot.goal_house import RobotRebootGoalHouse
from src.robot_reboot.state import RobotRebootState
from src.robot_reboot.util import join_quadrants, transpose_position_to_quadrant, build_matrix, \
    generate_positions_except


class RobotRebootFactory:
    """Factory to create Robot reboot games
    """

    def create(self, size, locate_robot_close_goal=False, n_movements=5, zobrist_hash_generator=None,
               move_all_robots=False):
        """Creates a robot reboot game and its initial state.
        The maze is randomly created joining different quadrants and the robots are randomly located in the maze
        avoiding the game goal house for all the robots.
        Args:
            size (int): size of the maze (number of rows and cols)
            locate_robot_close_goal (boolean): determine if the robot reaching its house should be located closer to its house
            n_movements: max number of movements away from its house if the robot is reallocated
            zobrist_hash_generator: if provided only unique states are allowed through out the game
        Returns:
            game: Robot reboot game
            state: Initial robot reboot game state
            index: quadrant's indexes selected to build the maze
        """
        game_configurations, n_robots = self.get_game_configurations(size)
        configurations_size = len(game_configurations)
        assert configurations_size > 0, "At least one configuration is needed to create the game's maze"
        index = np.arange(0, configurations_size)
        np.random.shuffle(index)
        c1 = game_configurations[index[0]]
        c2 = game_configurations[index[1 % configurations_size]]
        c3 = game_configurations[index[2 % configurations_size]]
        c4 = game_configurations[index[3 % configurations_size]]
        maze = join_quadrants(c1.quadrant, c2.quadrant, c3.quadrant, c4.quadrant)

        # Select a quadrant to put a house

        selected_quadrant_index = np.random.randint(0, high=4)
        conf = game_configurations[index[selected_quadrant_index % configurations_size]]
        houses = conf.houses
        selected_house_goal = houses[np.random.randint(0, high=len(houses))]
        x, y = transpose_position_to_quadrant(conf.quadrant, selected_house_goal.house, selected_quadrant_index + 1)

        pos = (x, y)
        goal = RobotRebootGoalHouse(selected_house_goal.robot_id, pos)
        game = RobotRebootGame(n_robots, maze, goal)
        robots_positions = self.move_robots_backwards_from_goal(game, goal, locate_robot_close_goal, n_movements,
                                                                n_robots, pos, size, move_all_robots=move_all_robots)
        state = RobotRebootState(game, robots_positions, zobrist_hash_generator=zobrist_hash_generator)
        return game, state, index[0:4]

    def move_robots_backwards_from_goal(self, game, goal, locate_robot_close_goal, max_movements, n_robots, pos, size,
                                        move_all_robots=False):
        robots_positions = list(generate_positions_except(n_robots, size, pos))
        if locate_robot_close_goal:
            robot = goal.robot_id
            goal_pos = goal.house
            robots_positions_2 = robots_positions.copy()
            robots_positions_2[robot] = goal_pos
            s = RobotRebootState(game, robots_positions_2)
            i = 0

            valid_actions = game.actions
            if not move_all_robots:
                valid_actions = [a for a in valid_actions if a.robot_id == robot]

            while i < max_movements or game.is_over(s):
                action = valid_actions[np.random.randint(0, len(valid_actions))]
                nex_state = game.apply(action, s)
                if s.robots_positions[robot] != nex_state.robots_positions[robot]:
                    i += 1
                    s = nex_state

            robots_positions = s.robots_positions
        return robots_positions

    def get_game_configurations(self, size):
        """Get all the game configurations for maze size
        Args:
            size (int): maze size
        Returns:
            confs (list): list of game configurations
        """
        if size == 11:
            confs = self.__get_5_x_5_quadrants()
        elif size == 31:
            confs = self.__get_15x_15_quadrants()
        else:
            raise UnsupportedMazeSize(size)
        return confs

    def __get_5_x_5_quadrants(self):
        """Quadrants for a 11x11 maze"""
        n = 5
        return [
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(2, 1), (3, 2)]), [RobotRebootGoalHouse(0, (2, 2))]
                   )
               ], 2

    def __get_15x_15_quadrants(self):
        "Quadrants for an 31x31 maze"
        n = 15
        yellow = 0
        green = 1
        red = 2
        blue = 3
        return [
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(0, 7), (4, 11), (5, 10), (7, 4), (8, 5), (9, 0), (10, 13), (11, 2),
                                        (11, 14),
                                        (12, 1), (13, 14), (14, 13)]),
                       [RobotRebootGoalHouse(blue, (4, 10)), RobotRebootGoalHouse(green, (8, 4)),
                        RobotRebootGoalHouse(red, (10, 14)), RobotRebootGoalHouse(yellow, (12, 2))]
                   ),
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(0, 7), (2, 1), (3, 2), (3, 12), (4, 13), (8, 5), (9, 4), (9, 14),
                                        (10, 13),
                                        (11, 0), (13, 14), (14, 13)]),
                       [RobotRebootGoalHouse(red, (2, 2)), RobotRebootGoalHouse(green, (4, 12)),
                        RobotRebootGoalHouse(blue, (8, 4)), RobotRebootGoalHouse(yellow, (10, 14))]
                   ),
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(0, 5), (2, 9), (3, 10), (7, 0), (8, 7), (9, 6), (9, 12), (10, 11),
                                        (11, 2), (12, 3), (13, 14), (14, 13)]),
                       [RobotRebootGoalHouse(blue, (2, 10)), RobotRebootGoalHouse(red, (8, 6)),
                        RobotRebootGoalHouse(green, (10, 12)), RobotRebootGoalHouse(yellow, (12, 2))]
                   ),
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(0, 9), (3, 12), (4, 11), (5, 4), (6, 5), (7, 6), (10, 3), (11, 2),
                                        (13, 0), (13, 14), (14, 13)]),
                       [RobotRebootGoalHouse(yellow, (4, 12)), RobotRebootGoalHouse(blue, (6, 4)),
                        RobotRebootGoalHouse(green, (6, 6)), RobotRebootGoalHouse(red, (10, 2))]
                   ),
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(0, 9), (2, 13), (3, 2), (3, 12), (4, 1), (9, 12), (10, 13), (11, 0),
                                        (12, 5), (13, 6), (13, 14), (14, 13)]),
                       [RobotRebootGoalHouse(yellow, (2, 12)), RobotRebootGoalHouse(green, (4, 2)),
                        RobotRebootGoalHouse(blue, (10, 12)), RobotRebootGoalHouse(red, (12, 6))]
                   ),
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(0, 3), (1, 8), (2, 7), (3, 2), (4, 3), (6, 13), (7, 12), (11, 0),
                                        (12, 5), (13, 6), (13, 14), (14, 13)]),
                       [RobotRebootGoalHouse(red, (2, 8)), RobotRebootGoalHouse(green, (4, 2)),
                        RobotRebootGoalHouse(yellow, (6, 12)), RobotRebootGoalHouse(blue, (12, 6))]
                   ),
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(0, 5), (3, 12), (4, 11), (5, 10), (10, 1), (11, 2), (13, 0), (13, 8),
                                        (13, 14), (14, 9), (14, 13)]),
                       [RobotRebootGoalHouse(green, (4, 10)), RobotRebootGoalHouse(yellow, (4, 12)),
                        RobotRebootGoalHouse(red, (10, 2)), RobotRebootGoalHouse(blue, (14, 8))]
                   ),
                   RobotRebootGameConfiguration(
                       build_matrix(n,
                                    [(0, 11), (3, 6), (4, 5), (6, 9), (7, 0), (7, 4), (7, 10), (8, 5), (10, 9),
                                     (11, 8), (13, 14), (14, 13)]),
                       [RobotRebootGoalHouse(yellow, (4, 6)), RobotRebootGoalHouse(blue, (6, 10)),
                        RobotRebootGoalHouse(red, (8, 4)), RobotRebootGoalHouse(green, (10, 8))]
                   ),
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(0, 9), (4, 13), (5, 0), (5, 12), (7, 4), (8, 5), (9, 6), (11, 10),
                                        (12, 9),
                                        (13, 14), (14, 13)]),
                       [RobotRebootGoalHouse(blue, (4, 12)), RobotRebootGoalHouse(green, (8, 4)),
                        RobotRebootGoalHouse(red, (8, 6)), RobotRebootGoalHouse(yellow, (12, 10))]
                   ),
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(0, 3), (1, 8), (2, 9), (6, 1), (7, 2), (9, 10), (10, 9), (11, 0),
                                        (12, 7), (13, 6), (13, 14), (14, 13)]),
                       [RobotRebootGoalHouse(green, (2, 8)), RobotRebootGoalHouse(red, (6, 2)),
                        RobotRebootGoalHouse(yellow, (10, 10)), RobotRebootGoalHouse(blue, (12, 6))]
                   ),
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(0, 9), (2, 11), (3, 12), (5, 2), (6, 3), (7, 10), (8, 9), (10, 5),
                                        (11, 4), (13, 0), (13, 14), (14, 13)]),
                       [RobotRebootGoalHouse(blue, (2, 12)), RobotRebootGoalHouse(yellow, (6, 2)),
                        RobotRebootGoalHouse(green, (8, 10)), RobotRebootGoalHouse(red, (10, 4))]
                   ),
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(0, 9), (2, 5), (3, 4), (6, 1), (7, 2), (7, 12), (8, 11), (9, 0),
                                        (11, 10), (12, 11), (13, 14), (14, 13)]),
                       [RobotRebootGoalHouse(red, (2, 4)), RobotRebootGoalHouse(green, (6, 2)),
                        RobotRebootGoalHouse(yellow, (8, 12)), RobotRebootGoalHouse(blue, (12, 10))]
                   ),
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(0, 3), (1, 6), (2, 5), (6, 13), (7, 12), (8, 1), (9, 2), (11, 8),
                                        (12, 9),
                                        (13, 0), (13, 14), (14, 13)]),
                       [RobotRebootGoalHouse(green, (2, 6)), RobotRebootGoalHouse(yellow, (6, 12)),
                        RobotRebootGoalHouse(red, (8, 2)), RobotRebootGoalHouse(blue, (12, 8))]
                   ),
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(0, 11), (5, 2), (6, 1), (8, 13), (9, 12), (11, 0), (11, 4), (12, 5),
                                        (13, 6), (13, 14), (14, 13)]),
                       [RobotRebootGoalHouse(red, (6, 2)), RobotRebootGoalHouse(yellow, (8, 12)),
                        RobotRebootGoalHouse(green, (12, 4)), RobotRebootGoalHouse(blue, (12, 6))]
                   ),
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(0, 9), (1, 4), (2, 3), (6, 11), (7, 12), (9, 0), (9, 8), (10, 9),
                                        (12, 3), (13, 2), (13, 14), (14, 13)]),
                       [RobotRebootGoalHouse(yellow, (2, 4)), RobotRebootGoalHouse(blue, (6, 12)),
                        RobotRebootGoalHouse(red, (10, 8)), RobotRebootGoalHouse(green, (12, 2))]
                   ),
                   RobotRebootGameConfiguration(
                       build_matrix(n, [(0, 7), (2, 11), (3, 10), (4, 1), (5, 2), (7, 0), (7, 12), (8, 11),
                                        (11, 4), (12, 5), (13, 14), (14, 13)]),
                       [RobotRebootGoalHouse(green, (2, 10)), RobotRebootGoalHouse(red, (4, 2)),
                        RobotRebootGoalHouse(yellow, (8, 12)), RobotRebootGoalHouse(blue, (12, 4))]
                   )
               ], 4
