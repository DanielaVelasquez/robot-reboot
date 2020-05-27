import pygame
import numpy as np

from models.robotreboot import RobotReboot
from models.maze import Maze
from models.robotreboot import Goal


class RobotView:
    def __init__(self, robot_id, colour=(0, 150, 0)):
        self.id = robot_id
        self.colour = colour


class RobotRebootView:
    def __init__(self, robotReboot, screen_size=(600, 600), run=True):
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

    def update(self, mode='human'):
        self.__view_update(mode)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.__select_robot(event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    self.__move_robot_on_game(RobotReboot.S)
                elif event.key == pygame.K_UP:
                    self.__move_robot_on_game(RobotReboot.N)
                elif event.key == pygame.K_RIGHT:
                    self.__move_robot_on_game(RobotReboot.E)
                elif event.key == pygame.K_LEFT:
                    self.__move_robot_on_game(RobotReboot.W)
            elif event.type == pygame.QUIT:
                self.quit_game()

            if self.robotReboot.done:
                self.quit_game()

    def __move_robot_on_game(self, direction):
        if self.__selected_robot is not None:
            self.robotReboot.move_robot(self.__selected_robot, direction)

    def __select_robot(self, mouse_position):
        maze_pos = self.__get_maze_position(mouse_position)
        self.__selected_robot = self.__find_robot(maze_pos)
        self.__move_robot = None

    def __find_robot(self, position):
        for robot_id in self.robotReboot.robots:
            robot_pos = self.robotReboot.robots[robot_id]
            if robot_pos == position:
                return robot_id
        return None

    def __get_maze_position(self, mouse_position):
        x = mouse_position[1]
        y = mouse_position[0]
        return int(x / self.cell_height), int(y / self.cell_width)

    def __view_update(self, mode='human'):
        if not self.robotReboot.done:
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
        for y in range(self.robotReboot.maze.height + 1):
            pygame.draw.line(self.maze_layer, line_colour, (0, y * self.cell_height),
                             (self.screen_width, y * self.cell_height))

        # Vertical lines
        for x in range(self.robotReboot.maze.width + 1):
            pygame.draw.line(self.maze_layer, line_colour, (x * self.cell_width, 0),
                             (x * self.cell_width, self.screen_height))

        # Draw walls
        cells = self.robotReboot.maze.cells.transpose()
        for x in range(len(cells)):
            for y in range(len(cells[x])):
                wall_status = self.robotReboot.maze.get_walls_status(cells[x, y])
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
            r = int(min(self.cell_width, self.cell_width) / 5 + 0.5)

            if robot_id in self.robots_view:
                colour = self.robots_view[robot_id].colour
            else:
                colour = tuple(np.random.choice(range(256), size=3))
                self.robots_view[robot_id] = RobotView(robot_id, colour)
            if self.robotReboot.is_goal_robot(robot_id):
                self.__goal_color = colour

            pygame.draw.circle(self.maze_layer, colour + (transparency,), (x, y), r)

    def __draw_goal(self, transparency=235):
        self.__colour_cell(self.robotReboot.goal.cell, self.__goal_color, transparency)

    def __colour_cell(self, cell, colour, transparency):
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
        return float(self.screen_width) / float(self.robotReboot.maze.width)

    @property
    def cell_height(self):
        return float(self.screen_height) / float(self.robotReboot.maze.height)


if __name__ == "__main__":
    data = np.load('./data/maze.npy')  # .transpose()
    np.random.seed(0)
    maze = Maze(data)
    goal = Goal("A", (0, 4))
    robots = {"A": (0, 0), "B": (4, 1), "C": (2, 2)}
    rr = RobotReboot(maze, robots, goal)
    rrView = RobotRebootView(rr)