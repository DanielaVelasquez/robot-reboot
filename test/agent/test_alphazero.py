import unittest

from src.agent.alphazero import AlphaZeroAgent
from src.experience.alphazero_experience import AlphaZeroExperienceCollector
from src.game_simulator.base import simulate_game
from test.robot_reboot.util import setup_and_get_encoder_state_model_for_robot_reboot_game


class TestAlphaZeroAgent(unittest.TestCase):

    def test_select_action_without_collector(self):
        encoder, game_state, model = setup_and_get_encoder_state_model_for_robot_reboot_game()
        alphazero_agent = AlphaZeroAgent(model, encoder, rounds_per_action=1)
        action = alphazero_agent.select_action(game_state)
        self.assertIsNotNone(action)
        next_state = game_state.apply(action)
        self.assertNotEqual(next_state, game_state)
        self.assertEquals(2, len(next_state.previous_states))

    def test_select_action_with_collector(self):
        encoder, game_state, model = setup_and_get_encoder_state_model_for_robot_reboot_game()
        collector = AlphaZeroExperienceCollector()
        collector.begin_episode()
        alphazero_agent = AlphaZeroAgent(model, encoder, rounds_per_action=1, collector=collector)
        action = alphazero_agent.select_action(game_state)
        self.assertIsNotNone(action)
        next_state = game_state.apply(action)
        collector.complete_episode(next_state.get_value())
        self.assertNotEqual(next_state, game_state)
        self.assertEquals(2, len(next_state.previous_states))

    def test_train_with_one_experience(self):
        encoder, game_state, model = setup_and_get_encoder_state_model_for_robot_reboot_game()
        collector = AlphaZeroExperienceCollector()
        alphazero_agent = AlphaZeroAgent(model, encoder, rounds_per_action=1, collector=collector)
        simulate_game(game_state, alphazero_agent, collector, max_actions=10)
        alphazero_agent.train(collector.to_buffer(), 0.01, 100)
