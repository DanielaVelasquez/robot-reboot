import unittest

from src.agent.alphazero import AlphaZeroAgent
from src.experience.alphazero_experience import AlphaZeroExperienceCollector
from src.game_simulator.base import simulate_game
from test.robot_reboot.util import setup_and_get_encoder_state_model_for_robot_reboot_game


class GameSimulator(unittest.TestCase):
    def test_simulate_game_with_one_action_max(self):
        encoder, game_state, model = setup_and_get_encoder_state_model_for_robot_reboot_game()
        alphazero_agent = AlphaZeroAgent(model, encoder, rounds_per_action=1)
        collector = AlphaZeroExperienceCollector()
        final_state = simulate_game(game_state, alphazero_agent, collector, max_actions=1)
        self.assertNotEqual(final_state, game_state)
        self.assertEquals(1, final_state.sequence_i)

    def test_simulate_game_without_max(self):
        encoder, game_state, model = setup_and_get_encoder_state_model_for_robot_reboot_game()
        alphazero_agent = AlphaZeroAgent(model, encoder, rounds_per_action=1)
        collector = AlphaZeroExperienceCollector()
        final_state = simulate_game(game_state, alphazero_agent, collector)
        self.assertNotEqual(final_state, game_state)
        self.assertNotEqual(1, final_state.sequence_i)
