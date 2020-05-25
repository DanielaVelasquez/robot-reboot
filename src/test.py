import numpy as np
from models.maze import Maze
from models.robotreboot import RobotReboot
from models.robotreboot import Goal
from robotreboot_view import RobotRebootView


def main():
    data = np.load('./data/maze.npy').transpose()
    np.random.seed(0)
    print(data)

    maze = Maze(data)
    goal= Goal("A", (0, 4))
    robots = {"A": (0, 0), "B": (4, 1), "C": (2, 2)}
    rr = RobotReboot(maze, robots, goal)
    rrView = RobotRebootView(rr)


if __name__ == '__main__':
    main()

# import pygame
# screen=pygame.display.set_mode([640, 480])
# screen.fill([255, 255, 255])
# red=255
# blue=0
# green=0
# left=50
# top=50
# width=90
# height=90
# filled=0
# pygame.draw.rect(screen, [red, blue, green], [left, top, width, height], filled)
# pygame.display.flip()
# running=True
# while running:
#     for event in pygame.event.get():
#         if event.type==pygame.QUIT:
#             running=False
# pygame.quit()