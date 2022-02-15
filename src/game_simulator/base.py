import sys


def simulate_game(game_state, agent, collector, max_actions=sys.maxsize):
    collector.begin_episode()
    actions_count = 0
    game = game_state.game
    while not game.is_over(game_state) and actions_count < max_actions:
        action = agent.select_action(game_state)
        game_state = game_state.apply(action)
        actions_count += 1
    collector.complete_episode(game_state.get_value())
    return game_state
