import numpy as np

maze = np.zeros((16, 16))

N = 1
E = 2
S = 4
W = 8

# First row
maze[0, 3] = E 
maze[0, 9] = E 
maze[0, 12] = S 

# Second row
maze[1, 6] = W
maze[1, 12] = W 
maze[1, 15] = S 

# Third row
maze[2, 1] = S 
maze[2, 6] = N 

#Forth row
maze[3, 1] = E 
maze[3, 8] = E 
maze[3, 9] = S 

# Fith row
maze[4, 4] = E 
maze[4, 5] = N 
maze[4, 14] = S 
maze[4, 15] = W 

#Sixth row
maze[5, 2] = S
maze[5, 3] = W
maze[5, 7] = S
maze[5, 8] = W
maze[5, 10] = N
maze[5, 11] = W

# Seven row
maze[6, 0] = S
maze[6, 7] = S
maze[6, 8] = S

# Eigth row
maze[7, 6] = E
maze[7, 9] = W

# Nine row
maze[8, 6] = E
maze[8, 7] = S
maze[8, 8] = S
maze[8, 9] = W

# Ten row
maze[9, 3] = N
maze[9, 4] = W
maze[9, 12] = E
maze[9, 13] = S

# Eleven row

# Twelve row
maze[11, 5] = E 
maze[11, 6] = N
maze[11, 9] = S
maze[11, 10] = W
maze[11, 15] = S

# Thirteen row
maze[12, 0] = E
maze[12, 1] = S

# Fourteen row
maze[13, 0] = S
maze[13, 14] = N
maze[13, 15] = W 

# Fifteen row
maze[14, 4] = S
maze[14, 5] = W
maze[14, 9] = E 
maze[14, 10] = N

# Sixteen row
maze[15, 6] = E
maze[15, 12] = W

maze = maze.astype(int)
print(maze)

np.save('maze_v1.npy', maze)
