import pygame
import numpy as np


class RobotRebootView:
    def __init__(self, robotReboot, screen_size=(600, 600)):
        pygame.init()
        pygame.display.set_caption("Robot Reboot")
        self.robotReboot = robotReboot
        # self.robotReboot.set_view(self)
        # Setting up screen
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(screen_size)
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill((255, 255, 255))
        # Layer for the maze
        self.maze_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
        self.maze_layer.fill((0, 0, 0, 0))

        self.__draw_maze()
        self.__draw_robots()
        self.__draw_goal()
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.maze_layer, (0, 0))
        pygame.display.flip()

    def __draw_maze(self):
        line_colour = (0, 0, 255, 15)

        # Horizontal lines
        for y in range(self.robotReboot.maze.height + 1):
            pygame.draw.line(self.maze_layer, line_colour, (0, y * self.cell_height),
                             (self.screen_width, y * self.cell_height))

        # Vertical lines
        for x in range(self.robotReboot.maze.width + 1):
            pygame.draw.line(self.maze_layer, line_colour, (x * self.cell_width, 0),
                             (x * self.cell_width, self.screen_height))
        #Draw walls
        for x in range(len(self.robotReboot.maze.cells)):
            for y in range(len(self.robotReboot.maze.cells[x])):
                wall_status = self.robotReboot.maze.get_walls_status(self.robotReboot.maze.cells[x, y])
                dirs = ""
                for dir, open in wall_status.items():
                    if open:
                        dirs += dir
                self.__cover_walls(x, y, dirs)

    def __cover_walls(self, x, y, dirs, colour=(0, 0, 0, 255)):
        dx = x * self.cell_width
        dy = y * self.cell_height

        for dir in dirs:
            if dir == "S":
                line_head = (dx + 1, dy + self.cell_height)
                line_tail = (dx + self.cell_width - 1, dy + self.cell_height)
            elif dir == "N":
                line_head = (dx + 1, dy)
                line_tail = (dx + self.cell_width - 1, dy)
            elif dir == "W":
                line_head = (dx, dy + 1)
                line_tail = (dx, dy + self.cell_height - 1)
            elif dir == "E":
                line_head = (dx + self.cell_height, dy + 1)
                line_tail = (dx + self.cell_width, dy + self.cell_height - 1)

            pygame.draw.line(self.maze_layer, colour, line_head, line_tail)

    def __draw_robots(self, transparency=255):
        for robot_id in self.robotReboot.robots:
            robot_pos = self.robotReboot.robots[robot_id]
            x = int(robot_pos[1] * self.cell_width + self.cell_width * 0.5 + 0.5)
            y = int(robot_pos[0] * self.cell_height + self.cell_height * 0.5 + 0.5)
            r = int(min(self.cell_width, self.cell_width)/5 + 0.5)
            colour = tuple(np.random.choice(range(256), size=3))
            if self.robotReboot.is_goal_robot(robot_id):
                self.__goal_color = colour
            pygame.draw.circle(self.maze_layer, colour + (transparency,), (x, y), r)

    def __draw_goal(self, transparency=235):
        self.__colour_cell(self.robotReboot.goal.cell, self.__goal_color, transparency)

    def __colour_cell(self, cell, colour, transparency):
        x = int(cell[1] * self.cell_width + 0.5 + 1)
        y = int(cell[0] * self.cell_height + 0.5 + 1)
        w = int(self.cell_width + 0.5 - 1)
        h = int(self.cell_height + 0.5 -1)
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
        return float(self.screen_width) / float(self.robotReboot.maze.width)

    @property
    def cell_height(self):
        return float(self.screen_height) / float(self.robotReboot.maze.height)
