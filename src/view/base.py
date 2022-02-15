import sys


def simulate_game_with_view(agent, game, game_state, view, collector=None, actions_max=sys.maxsize):
    if collector:
        collector.begin_episode()
    actions_count = 0
    while not game.is_over(game_state) and actions_count < actions_max:
        view.display(game_state)
        action = agent.select_action(game_state)
        game_state = game.apply(action, game_state)
        print(action)
        actions_count+=1
    print('Game is over. Score ', str(game.get_score(game_state)))
    if collector:
        collector.complete_episode(game_state.get_value())
    view.display(game_state)
