def simulate_game(agent, game, game_state, view, collector=None):
    if collector:
        collector.begin_episode()
    while not game.is_over(game_state):
        view.display(game_state)
        action = agent.select_action(game_state)
        game_state = game.apply(action, game_state)
        print(action)
    print('Game is over. Score ', str(game.get_score(game_state)))
    if collector:
        collector.complete_episode(game_state.get_value())
    view.display(game_state)
