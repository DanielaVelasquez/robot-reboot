from time import sleep

import numpy as np
import pygame

from src.robot_reboot.factory import RobotRebootFactory
from src.robot_reboot.maze_cell_type import MazeCellType
from src.ui.base import GameStateView


class RobotView:
    def __init__(self, robot_id, colour=(0, 150, 0)):
        self.id = robot_id
        self.colour = colour


def get_color_based_on_name(robot_id):
    if robot_id == 0:
        return tuple([252, 219, 3])
    elif robot_id == 1:
        return tuple([3, 252, 161])
    elif robot_id == 2:
        return tuple([3, 219, 252])
    elif robot_id == 3:
        return tuple([224, 76, 76])
    else:
        return tuple(np.random.choice(range(256), size=3))


class RobotRebootGameView(GameStateView):
    def __init__(self, state, game, movements: list, screen_size=(600, 600), run=True):
        pygame.init()
        pygame.display.set_caption("Robot Reboot")
        self.state = state
        self.game = game
        self.movements = movements
        self.movements_index = 0
        # Setting up screen
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(screen_size)
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill((255, 255, 255))
        # Layer for the maze
        self.maze_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
        self.maze_layer.fill((0, 0, 0, 0))

        self.robots_view = {}
        self.__selected_robot = None

        self.__draw_maze()
        self.__draw_robots()
        self.__draw_goal()
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.maze_layer, (0, 0))
        if run:
            pygame.display.flip()
            while run:
                self.update()
                sleep(1)

    def update(self, mode='human'):
        self.__view_update(mode)
        if self.movements_index < len(self.movements):
            m = self.movements.__getitem__(self.movements_index)
            print(m)
            print(f'Movement {self.movements_index}/{len(self.movements)}')
            self.state.move_robot(m[0], m[1])
            self.movements_index += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_game()

    def __move_robot_on_game(self, direction):
        if self.__selected_robot is not None:
            self.state.move_robot(self.__selected_robot, direction)

    def __select_robot(self, mouse_position):
        maze_pos = self.__get_maze_position(mouse_position)
        self.__selected_robot = self.__find_robot(maze_pos)
        self.__move_robot = None

    def __find_robot(self, position):
        for robot_id in self.state.robots:
            robot_pos = self.state.robots[robot_id]
            if robot_pos == position:
                return robot_id
        return None

    def __get_maze_position(self, mouse_position):
        x = mouse_position[1]
        y = mouse_position[0]
        return int(x / self.cell_height), int(y / self.cell_width)

    def __view_update(self, mode='human'):
        self.maze_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
        self.maze_layer.fill((0, 0, 0, 0))
        self.__draw_maze()
        self.__draw_robots()
        self.__draw_goal()
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.maze_layer, (0, 0))
        if mode == "human":
            pygame.display.flip()

        return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

    def __draw_maze(self):
        line_colour = (0, 0, 255, 15)

        # Horizontal lines
        rows, cols = self.game.maze.shape
        for y in range(rows + 1):
            pygame.draw.line(self.maze_layer, line_colour, (0, y * self.cell_height),
                             (self.screen_width, y * self.cell_height))

        # Vertical lines
        for x in range(cols + 1):
            pygame.draw.line(self.maze_layer, line_colour, (x * self.cell_width, 0),
                             (x * self.cell_width, self.screen_height))

        # Draw walls
        cells = self.game.maze.transpose()
        for x in range(len(cells)):
            for y in range(len(cells[x])):
                if cells[x, y] == MazeCellType.WALL.value:
                    self.__colour_cell((x, y), tuple([224, 76, 76]))

    def __draw_robots(self, transparency=255):
        positions = self.state.robots_positions
        for robot_id in range(len(positions)):
            robot_pos = positions[robot_id]
            x = int(robot_pos[1] * self.cell_width + self.cell_width * 0.5 + 0.5)
            y = int(robot_pos[0] * self.cell_height + self.cell_height * 0.5 + 0.5)
            r = int(min(self.cell_width, self.cell_width) / 5 + 0.5)

            if robot_id in self.robots_view:
                colour = self.robots_view[robot_id].colour
            else:
                colour = get_color_based_on_name(robot_id)
                self.robots_view[robot_id] = RobotView(robot_id, colour)
            if self.game.goal_house.robot_id == robot_id:
                self.__goal_color = colour

            pygame.draw.circle(self.maze_layer, colour + (transparency,), (x, y), r)

    def __draw_goal(self, transparency=235):
        self.__colour_cell(self.game.goal_house.house, self.__goal_color, transparency)

    def __colour_cell(self, cell, colour, transparency=235):
        x = int(cell[1] * self.cell_width + 0.5 + 1)
        y = int(cell[0] * self.cell_height + 0.5 + 1)
        w = int(self.cell_width + 0.5 - 1)
        h = int(self.cell_height + 0.5 - 1)
        pygame.draw.rect(self.maze_layer, colour + (transparency,), (x, y, w, h))

    def quit_game(self):
        pygame.display.quit()
        pygame.quit()

    @property
    def screen_width(self):
        return int(self.screen_size[0])

    @property
    def screen_height(self):
        return int(self.screen_size[1])

    @property
    def cell_width(self):
        return float(self.screen_width) / float(self.game.maze.shape[1])

    @property
    def cell_height(self):
        return float(self.screen_height) / float(self.game.maze.shape[0])


if __name__ == "__main__":
    factory = RobotRebootFactory()
    game, state, selected_quadrants = factory.create(31)
    rrView = RobotRebootGameView(state, game, list())
