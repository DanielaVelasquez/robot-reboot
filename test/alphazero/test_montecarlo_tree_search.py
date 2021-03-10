import unittest
from unittest.mock import Mock

import numpy as np

from exceptions.alphazero.monte_carlo_tree_search import InvalidDepthException, InvalidPlayoutException
from exceptions.exceptions import RequiredValueException
from src.alphazero.game_player import GamePlayer
from src.alphazero.heuristic_function import heuristic_fn
from src.alphazero.montecarlo_tree_search import MonteCarloTreeSearch
from src.robot_reboot.factory import RobotRebootFactory
from src.robot_reboot.game import RobotRebootGame
from src.robot_reboot.goal_house import RobotRebootGoalHouse
from src.robot_reboot.state import RobotRebootState
from test.alphazero.fake_data import FakeState, FakeModel, fn_predict_probability_1_for_next_action, FakeGame


def assert_state(mcts, s, n=[], w=[], p=[], message=""):
    np.testing.assert_equal(mcts.states_statistics[s].n, n, message)
    np.testing.assert_equal(mcts.states_statistics[s].w, w, message)
    np.testing.assert_equal(mcts.states_statistics[s].p, p, message)


def fake_heuristic_fn(p, _):
    return p


def fn_predict_probability_np_seed(actions_size, state):
    """Fake predict probabilities of winning over each action based on an uniform distribution
    with values between [-1, 1]
    (Function used in the FakeModel)
    Args:
       actions_size (int):   number of actions
        state       (State): state to calculate the probabilities of winning per actions
    Returns:
        p (np array): probality over actions with an uniform distribution.
    """
    return np.random.uniform(low=-1, high=1, size=actions_size)


class TestMonteCarloTreeSearch(unittest.TestCase):

    def test_init_passes_without_default_parameters(self):
        mock_game_player = Mock()
        MonteCarloTreeSearch(Mock(), 1, mock_game_player)

    def test_init_passes_with_default_parameters(self):
        mock_game_player = Mock()
        MonteCarloTreeSearch(Mock(), 1, mock_game_player, playouts=1)

    def test_init_fails_when_heuristic_fn_is_none(self):
        mock_game_player = Mock()
        self.assertRaises(RequiredValueException, lambda: MonteCarloTreeSearch(None, 1, mock_game_player))

    def test_init_fails_when_max_depth_is_zero(self):
        mock_game_player = Mock()
        self.assertRaises(InvalidDepthException, lambda: MonteCarloTreeSearch(Mock(), 0, mock_game_player))

    def test_init_fails_when_max_depth_is_below_zero(self):
        mock_game_player = Mock()
        self.assertRaises(InvalidDepthException, lambda: MonteCarloTreeSearch(Mock(), -1, mock_game_player))

    def test_init_fails_when_game_player_none(self):
        self.assertRaises(RequiredValueException, lambda: MonteCarloTreeSearch(Mock(), 1, None))

    def test_init_fails_when_playouts_is_zero(self):
        mock_game_player = Mock()
        self.assertRaises(InvalidPlayoutException,
                          lambda: MonteCarloTreeSearch(Mock(), 1, mock_game_player, playouts=0))

    def test_init_fails_when_playouts_is_below_zero(self):
        mock_game_player = Mock()
        self.assertRaises(InvalidPlayoutException,
                          lambda: MonteCarloTreeSearch(Mock(), 1, mock_game_player, playouts=-1))

    def test_search_with_depth_2(self):
        """
                        Tree built
                            s0

                (a1)    (a2)    (a3)    (a4)
                 s1      s2      s3      s1

                (a2)    (a3)            (a1)
                 s2      s3              s1

               v=-1      v=1     v=1     v=0

        """
        fake_model = FakeModel(fn_predict_probability_1_for_next_action, FakeGame())
        game_player = GamePlayer(fake_model, fake_model.game)
        mcts = MonteCarloTreeSearch(fake_heuristic_fn, 2, game_player, playouts=1)
        fake_state = FakeState(game_player.game, 0, 0)
        p = mcts.search(fake_state)
        np.testing.assert_equal(p, [-1, 1, 1, 0])
        self.assertEqual(list(mcts.states_statistics.keys()), ['s1', 's2', 's4'])
        assert_state(mcts, 's1', n=[0, 1, 0, 0], w=[0, -1, 0, 0], p=[0, -1, 0, 0])
        assert_state(mcts, 's2', n=[0, 0, 1, 0], w=[0, 0, 1, 0], p=[0, 0, 1, 0])
        assert_state(mcts, 's4', n=[1, 0, 0, 0], w=[0, 0, 0, 0], p=[0, 0, 0, 0])

    def test_search_with_all_leaves_win(self):
        """
                                Tree built
                                    s0

                        (a1)    (a2)    (a3)    (a4)
                         s1      s2      s3      s1

                        (a2)    (a3)            (a1)
                         s2      s3              s1

                        (a3)                    (a2)
                         s3                      s2

                                                (a3)
                                                 s3
                        v=1      v=1     v=1     v=1

        """

        fake_model = FakeModel(fn_predict_probability_1_for_next_action, FakeGame())
        game_player = GamePlayer(fake_model, fake_model.game)
        mcts = MonteCarloTreeSearch(fake_heuristic_fn, 4, game_player, playouts=1)
        fake_state = FakeState(game_player.game, 0, 0)
        p = mcts.search(fake_state)
        np.testing.assert_equal(p, [1, 1, 1, 1])
        self.assertEqual(list(mcts.states_statistics.keys()), ['s1', 's2', 's4'])
        assert_state(mcts, 's1', n=[0, 2, 0, 0], w=[0, 2, 0, 0], p=[0, 1, 0, 0])
        assert_state(mcts, 's2', n=[0, 0, 3, 0], w=[0, 0, 3, 0], p=[0, 0, 1, 0])
        assert_state(mcts, 's4', n=[1, 0, 0, 0], w=[1, 0, 0, 0], p=[1, 0, 0, 0])

    def test_search_clean_state_statistics_for_each_search(self):
        """
                               Tree built
                                   s0

                       (a1)    (a2)    (a3)    (a4)
                        s1      s2      s3      s1

                       (a2)    (a3)            (a1)
                        s2      s3              s1

                       v=-1      v=1     v=1     v=0

       """
        fake_model = FakeModel(fn_predict_probability_1_for_next_action, FakeGame())
        game_player = GamePlayer(fake_model, fake_model.game)
        mcts = MonteCarloTreeSearch(fake_heuristic_fn, 2, game_player, playouts=1)
        fake_state = FakeState(game_player.game, 0, 0)
        # Executed twice, visits should not be added on top of the second search
        mcts.search(fake_state)
        p = mcts.search(fake_state)

        np.testing.assert_equal(p, [-1, 1, 1, 0])

        self.assertEqual(list(mcts.states_statistics.keys()), ['s1', 's2', 's4'])
        assert_state(mcts, 's1', n=[0, 1, 0, 0], w=[0, -1, 0, 0], p=[0, -1, 0, 0],
                     message="Visits for s1 should only count one the last search, check state statistics are reset")
        assert_state(mcts, 's2', n=[0, 0, 1, 0], w=[0, 0, 1, 0], p=[0, 0, 1, 0],
                     message="Visits for s2 should only count one the last search, check state statistics are reset")
        assert_state(mcts, 's4', n=[1, 0, 0, 0], w=[0, 0, 0, 0], p=[0, 0, 0, 0],
                     message="Visits for s3 should only count one the last search, check state statistics are reset")

    def test_search_with_random_probabilities(self):
        """
                                                    Tree built
                                                        s0

                    (a1)                        (a2)            (a3)                    (a4)
                     s1                          s2              s3                     s4

            (a4)    (a1)    (a1)        (a1)    (a1)    (a2)                    (a1)    (a4)    (a1)
             s4      s1      s1          s1      s1      s2                     s1      s4      s1

            (a1)    (a2)    (a3)        (a4)    (a3)    (a1)                    (a3)   (a4)    (a3)
             s1      s2      s3          s4      s3      s1                      s3     s4      s3

            v=0      v=-1    v=1         v=-1    v=1     v=0    v=1 v=1 v=1      v=1    v=-1    v=0

        """
        np.random.seed(26)
        fake_model = FakeModel(fn_predict_probability_np_seed, FakeGame())
        game_player = GamePlayer(fake_model, fake_model.game)
        mcts = MonteCarloTreeSearch(fake_heuristic_fn, 3, game_player, playouts=3)
        fake_state = FakeState(game_player.game, 0, 0)
        p = mcts.search(fake_state)
        np.testing.assert_equal(p, [0, 0, 1, 0])
        self.assertEqual(sorted(mcts.states_statistics.keys()), sorted(['s1', 's2', 's4']))
        assert_state(mcts, 's1', n=[3, 1, 3, 2], w=[0, -1, 3, -1], p=[0, -1, 1, -0.5])
        assert_state(mcts, 's2', n=[3, 1, 0, 0], w=[0, 0, 0, 0], p=[0, 0, 0, 0])
        assert_state(mcts, 's4', n=[3, 0, 0, 2], w=[1, 0, 0, -2], p=[1 / 3, 0, 0, -1])

    def test_search_returns_all_probabilities_zero_when_none_valid_actions_on_root_state(self):
        """
        Given that there are no valid actions on the root state
        when MCTS searches
        Then all probabilities are zero because it can't explore the tree
        """
        fake_model = FakeModel(fn_predict_probability_1_for_next_action, FakeGame())
        fake_game = fake_model.game
        game_player = GamePlayer(fake_model, fake_model.game)
        mcts = MonteCarloTreeSearch(fake_heuristic_fn, 3, game_player, playouts=1)
        fake_state = FakeState(fake_game, 0, 0)
        # If state not defined here, all actions are returned
        fake_game.valid_state_actions_dict = {
            0: []
        }
        p = mcts.search(fake_state)
        np.testing.assert_equal(p, [0, 0, 0, 0])
        self.assertEqual(0, len(mcts.states_statistics))

    def test_search__when_none_valid_actions_on_next_states_after_root_state(self):
        """
        Given that all states after the root state don't have valid actions
        When MCTS searches
        Then no states were visited from the following states after the root state

                                        Tree built
                                            s0

                                (a1)    (a2)    (a3)    (a4)
                                 s1      s2      s3      s1

                                v=0      v=-1     v=1     v=-1

        """
        fake_model = FakeModel(fn_predict_probability_1_for_next_action, FakeGame())
        fake_game = fake_model.game
        game_player = GamePlayer(fake_model, fake_model.game)
        mcts = MonteCarloTreeSearch(fake_heuristic_fn, 3, game_player, playouts=1)
        fake_state = FakeState(fake_game, 0, 0)
        # If state not defined here, all actions are returned
        fake_game.valid_state_actions_dict = {
            1: [],
            2: [],
            3: [],
            4: []
        }
        p = mcts.search(fake_state)
        np.testing.assert_equal(p, [0, -1, 1, -1])
        self.assertEqual(0, len(mcts.states_statistics))

    def test_search__when_some_valid_actions_on_next_states(self):
        """
        Given that the s2 can't take action 3
        When MCTS searches
        Then s2 always takes action 4 because its the next best action
                                Tree built
                                    s0

                        (a1)    (a2)    (a3)    (a4)
                         s1      s2      s3      s1

                        (a2)    (a4)            (a1)
                         s2      s4              s1

                        (a4)    (a1)            (a2)
                         s4      s1              s2

                        v=-1     v=0     v=1     v=-1

        """
        fake_model = FakeModel(fn_predict_probability_1_for_next_action, FakeGame())
        fake_game = fake_model.game
        game_player = GamePlayer(fake_model, fake_model.game)
        mcts = MonteCarloTreeSearch(fake_heuristic_fn, 3, game_player, playouts=1)
        fake_state = FakeState(fake_game, 0, 0)
        # If state not defined here, all actions are returned
        fake_game.valid_state_actions_dict = {
            2: [a for a in fake_game.actions if a.value != 3],  # It can't go to state 3
        }
        p = mcts.search(fake_state)
        np.testing.assert_equal([-1, 0, 1, -1], p)
        self.assertEqual(sorted(mcts.states_statistics.keys()), sorted(['s1', 's2', 's4']))
        assert_state(mcts, 's1', n=[0, 2, 0, 0], w=[0, -2, 0, 0], p=[0, -1, 0, 0])
        assert_state(mcts, 's2', n=[0, 0, 0, 2], w=[0, 0, 0, -1], p=[0, 0, 0, -0.5])
        assert_state(mcts, 's4', n=[2, 0, 0, 0], w=[-1, 0, 0, 0], p=[-0.5, 0, 0, 0])

    def test_search__when_some_valid_actions_on_next_states_and_root_state(self):
        """
        Given that s0 can't take action 1 and s2 can't take action 3
        When MCTS searches
        Then s0 doesn't explore s1 at the root level
             s2 always takes action 4 because its the next best action

                            Tree built
                                s0

                        (a2)    (a3)    (a4)
                        s2      s3      s1

                        (a4)            (a1)
                         4              s1

                        (a1)            (a2)
                         s1              s2

                        v=0     v=1     v=-1

        """
        fake_model = FakeModel(fn_predict_probability_1_for_next_action, FakeGame())
        fake_game = fake_model.game
        game_player = GamePlayer(fake_model, fake_model.game)
        mcts = MonteCarloTreeSearch(fake_heuristic_fn, 3, game_player, playouts=1)
        fake_state = FakeState(fake_game, 0, 0)
        # If state not defined here, all actions are returned
        fake_game.valid_state_actions_dict = {
            0: [a for a in fake_game.actions if a.value != 1],  # It can't go to state 1
            2: [a for a in fake_game.actions if a.value != 3],  # It can't go to state 3
        }
        p = mcts.search(fake_state)
        np.testing.assert_equal([0, 0, 1, -1], p)
        self.assertEqual(sorted(mcts.states_statistics.keys()), sorted(['s1', 's2', 's4']))
        assert_state(mcts, 's1', n=[0, 1, 0, 0], w=[0, -1, 0, 0], p=[0, -1, 0, 0])
        assert_state(mcts, 's2', n=[0, 0, 0, 1], w=[0, 0, 0, 0], p=[0, 0, 0, 0])
        assert_state(mcts, 's4', n=[2, 0, 0, 0], w=[-1, 0, 0, 0], p=[-0.5, 0, 0, 0])

    def test_search_with_robot_reboot_game(self):
        """
        Given there is one robot in position (1,1) of the maze,
              its goal is cell (0,0),
              np seed is 26,
              actions order is N, S, E, W
        When MCTS searches
        The tree generated is
                            Tree built
                              (1,1)

                        (N)       (E)        (W)
                       (0,1)     (1,2)      (1,0)

                        (W)       (N)        (E)
                       (0,0)     (0,2)      (1,2)

                                  (W)        (N)
                                 (0,0)      (0,2)

                        v=1       v=1       v=-0

        """
        np.random.seed(26)
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0],
                         [0, 0, 0],
                         [0, 1, 0],
                         ])
        game = RobotRebootGame(1, maze, house)
        fake_model = FakeModel(fn_predict_probability_np_seed, game)
        game_player = GamePlayer(fake_model, game)
        mcts = MonteCarloTreeSearch(fake_heuristic_fn, 3, game_player, playouts=1)
        s = RobotRebootState(game, [(1, 1)])
        p = mcts.search(s)
        np.testing.assert_equal([1, 1, 0, 0], p)

    def test_search_with_robot_reboot_game_using_factory_and_heuristic_fn(self):
        f = RobotRebootFactory(seed=26)

        game, state, _ = f.create(11)
        fake_model = FakeModel(fn_predict_probability_np_seed, game)
        game_player = GamePlayer(fake_model, game)
        mcts = MonteCarloTreeSearch(heuristic_fn, 3, game_player, playouts=1)
        np.str(game.maze)
        mcts.search(state)
