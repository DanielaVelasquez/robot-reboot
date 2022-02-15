def simulate_game(agent, game, game_state, view):
    while not game.is_over(game_state):
        view.display(game_state)
        action = agent.select_action(game_state)
        game_state = game.apply(action, game_state)
        print(action)
    print('Game is over. Score ', str(game.get_score(game_state)))
    view.display(game_state)
