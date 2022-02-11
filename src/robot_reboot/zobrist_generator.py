import random

MAX63 = 0x7fffffffffffffff


def generate_zobrist_hash(n_robots, rows, cols):
    maze = {}
    empty_board = 0
    for row in range(rows):
        for col in range(cols):
            if row %2 == 0 and col % 2 ==0 :
                for robot in range(n_robots):
                    code = random.randint(0, MAX63)
                    maze[(row, col), robot] = code

    print('    HASH_CODE = {')
    for (pt, robot), hash_code in maze.items():
        print('        (%r, %s): %r,' % (pt, robot, hash_code))
    print('    }')
    print('')
    print('EMPTY_BOARD = %d' % (empty_board,))


generate_zobrist_hash(4, 31, 31)
