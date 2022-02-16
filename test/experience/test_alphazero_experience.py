import os
import unittest

import h5py
import numpy as np

from src.agent.alphazero import AlphaZeroAgent
from src.experience.alphazero_experience import AlphaZeroExperienceCollector, load_experience
from src.game_simulator.base import simulate_game
from test.robot_reboot.util import setup_and_get_encoder_state_model_for_robot_reboot_game


def create_experience():
    encoder, game_state, model = setup_and_get_encoder_state_model_for_robot_reboot_game()
    collector = AlphaZeroExperienceCollector()
    alphazero_agent = AlphaZeroAgent(model, encoder, rounds_per_action=1, collector=collector)
    simulate_game(game_state, alphazero_agent, collector, max_actions=1)

    return collector


class TestAlphaZeroExperience(unittest.TestCase):

    def test_serialize_buffer(self):
        hdf5_filename = 'test_serialize_buffer.hdf5'
        collector = create_experience()
        buffer = collector.to_buffer()
        with h5py.File(hdf5_filename, 'w') as test_serialize_buffer:
            buffer.serialize(test_serialize_buffer)

        os.remove(hdf5_filename)

    def test_load_experience(self):
        hdf5_filename = 'test_load_experience'
        collector = create_experience()

        buffer = collector.to_buffer()
        with h5py.File(hdf5_filename, 'w') as test_serialize_buffer:
            buffer.serialize(test_serialize_buffer)

        with h5py.File(hdf5_filename, 'r') as test_read:
            read_experience = load_experience(test_read)

        np.testing.assert_equal(np.array(collector.states), read_experience.states)
        np.testing.assert_equal(np.array(collector.visit_counts), read_experience.visit_counts)
        np.testing.assert_equal(np.array(collector.rewards), read_experience.rewards)

        os.remove(hdf5_filename)
