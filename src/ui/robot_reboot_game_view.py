from time import sleep

import numpy as np
import pygame

from src.agent.human import HumanAgent
from src.robot_reboot.factory import RobotRebootFactory
from src.robot_reboot.maze_cell_type import MazeCellType
from src.robot_reboot.util import is_even
from src.ui.base import GameStateView


class RobotView:
    def __init__(self, robot_id, colour=(0, 150, 0)):
        self.id = robot_id
        self.colour = colour


def get_color_based_on_name(robot_id):
    if robot_id == 0:  # Yellow
        return tuple([252, 219, 3])
    elif robot_id == 1:  # Green
        return tuple([3, 252, 161])
    elif robot_id == 2:  # Blue
        return tuple([3, 219, 252])
    elif robot_id == 3:  # Red
        return tuple([224, 76, 76])
    else:
        return tuple(np.random.choice(range(256), size=3))


class RobotRebootGameView(GameStateView):
    def __init__(self, game, screen_size=(600, 600)):
        GameStateView.__init__(self)
        self.game = game
        self.__maze_layer = None
        self.__background = None
        self.__state = None
        # Setting up screen
        self.screen_size = screen_size
        self.__screen = pygame.display.set_mode(screen_size)
        self.__robots_view = {}
        self.__next_move = False
        pygame.init()
        pygame.display.set_caption("Robot Reboot")

    def display(self, state):
        self.__state = state
        self.__background = pygame.Surface(self.__screen.get_size()).convert()
        self.__background.fill((255, 255, 255))
        # Layer for the maze
        self.__maze_layer = pygame.Surface(self.__screen.get_size()).convert_alpha()
        self.__maze_layer.fill((0, 0, 0, 0))
        self.__draw_maze()
        self.__draw_robots()
        self.__draw_goal()
        self.__screen.blit(self.__background, (0, 0))
        self.__screen.blit(self.__maze_layer, (0, 0))
        pygame.display.flip()
        self.__next_move = False
        while not self.__next_move:
            self.update()

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_game()
            elif event.type == pygame.KEYUP:
                self.__next_move = True

    def __view_update(self):
        self.__maze_layer = pygame.Surface(self.__screen.get_size()).convert_alpha()
        self.__maze_layer.fill((0, 0, 0, 0))
        self.__draw_maze()
        self.__draw_robots()
        self.__draw_goal()
        self.__screen.blit(self.__background, (0, 0))
        self.__screen.blit(self.__maze_layer, (0, 0))
        return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

    def __draw_maze(self):
        line_colour = (0, 0, 255, 15)

        # Horizontal lines
        rows, cols = self.game.maze.shape
        for y in range(rows + 1):
            pygame.draw.line(self.__maze_layer, line_colour, (0, y * self.cell_height),
                             (self.screen_width, y * self.cell_height))

        # Vertical lines
        for x in range(cols + 1):
            pygame.draw.line(self.__maze_layer, line_colour, (x * self.cell_width, 0),
                             (x * self.cell_width, self.screen_height))

        # Draw walls
        cells = self.game.maze
        for x in range(len(cells)):
            for y in range(len(cells[x])):
                if cells[x, y] == MazeCellType.WALL.value:
                    self.__colour_cell((x, y), tuple([0, 0, 0]))
                elif not is_even(x) or not is_even(y):
                    self.__colour_cell((x, y), tuple([200, 200, 200]))

    def __draw_robots(self, transparency=255):
        positions = self.__state.robots_positions
        for robot_id in range(len(positions)):
            robot_pos = positions[robot_id]
            x = int(robot_pos[1] * self.cell_width + self.cell_width * 0.5 + 0.5)
            y = int(robot_pos[0] * self.cell_height + self.cell_height * 0.5 + 0.5)
            r = int(min(self.cell_width, self.cell_width) / 5 + 0.5)

            if robot_id in self.__robots_view:
                colour = self.__robots_view[robot_id].colour
            else:
                colour = get_color_based_on_name(robot_id)
                self.__robots_view[robot_id] = RobotView(robot_id, colour)
            if self.game.goal_house.robot_id == robot_id:
                self.__goal_color = colour

            pygame.draw.circle(self.__maze_layer, colour + (transparency,), (x, y), r)

    def __draw_goal(self, transparency=235):
        self.__colour_cell(self.game.goal_house.house, self.__goal_color, transparency)

    def __colour_cell(self, cell, colour, transparency=235):
        x = int(cell[1] * self.cell_width + 0.5 + 1)
        y = int(cell[0] * self.cell_height + 0.5 + 1)
        w = int(self.cell_width + 0.5 - 1)
        h = int(self.cell_height + 0.5 - 1)
        pygame.draw.rect(self.__maze_layer, colour + (transparency,), (x, y, w, h))

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
