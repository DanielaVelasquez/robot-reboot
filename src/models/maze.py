class Maze:
    '''
    Walls meaning
    '''
    N = 1
    E = 2
    S = 4
    W = 8

    def __init__(self, cells):
        self.cells = cells
        self.size = cells.shape

    @classmethod
    def get_walls_status(cls, cell):
        walls = {
            "N": (cell & 0x1) >> 0,
            "E": (cell & 0x2) >> 1,
            "S": (cell & 0x4) >> 2,
            "W": (cell & 0x8) >> 3,
        }
        return walls

    @property
    def height(self):
        return int(self.size[0])

    @property
    def width(self):
        return int(self.size[1])
