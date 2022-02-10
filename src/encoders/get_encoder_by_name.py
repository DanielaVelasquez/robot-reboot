import importlib


def get_encoder_by_name(name, maze_size):
    if isinstance(maze_size, int):
        maze_size = (maze_size, maze_size)
    module = importlib.import_module('src.encoders.' + name)
    constructor = getattr(module, 'create')
    return constructor(maze_size)
