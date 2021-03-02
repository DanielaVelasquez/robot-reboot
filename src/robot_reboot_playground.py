from alphazero.robot_reboot_factory import RobotRebootFactory
from alphazero.util import calculate_size_with_walls
from alphazero.deep_heuristic import DeepHeuristic
from alphazero.monte_carlo_tree_search import MonteCarloTreeSearch

if __name__ == "__main__":
    size = 8
    extended_size = calculate_size_with_walls(size)
    factory = RobotRebootFactory(size=size)
    victories = 0
    runs = 1
    avg_execution_time = 0
    # nn.save_model()
    for i in range(runs):
        game = factory.build()

        nn = DeepHeuristic(game.observation().shape, 8, model_name='model_8_x_8.h5')
        # nn.load_model()
        print(f'Game {i}')
        print(f'Robots = {game.robots}')
        print(f'Goal = {game.goal_house}')
        mcts = MonteCarloTreeSearch(nn, 0.5, 10)
        # Play that game until it is over
        actions_taken = 0
        while not game.is_over() and actions_taken < 10:
            action = mcts.best_action(game)
            # stop = timeit.default_timer()
            print(f'Game {i} with action {action}')
            game.move(action)
            actions_taken += 1
        print('Won' if game.score() == 1 else 'Lost')
        victories += game.score()

    print(f'Victories total {victories}')
    print(f'Average execution time {avg_execution_time / runs}')
