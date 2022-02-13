from src.game.game import Game


def get_map_actions_to_index_and_actions_list(game: Game):
    actions = game.actions
    return {actions[i]: i for i in range(len(actions))}, actions
