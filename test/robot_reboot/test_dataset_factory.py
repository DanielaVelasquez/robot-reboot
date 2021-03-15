import unittest

import numpy as np
from src.ml.model import get_cnn_model
from src.robot_reboot.dataset_factory import RobotRebootDataSetFactory


class TestDatasetFactory(unittest.TestCase):
    def test_create(self):
        cnn = get_cnn_model((31, 31, 9), n_outputs=16, convolutions=3, optimizer='adam', seed=26)
        ds_factory = RobotRebootDataSetFactory(31, cnn, max_depth=10, seed=26, playouts=1)
        v, p, s = ds_factory.create(locate_robot_close_goal=False, max_movements=5)
        self.assertIsNotNone(v)
        self.assertIsNotNone(p)
        self.assertIsNotNone(s)

    def test_create_different_state_everytime_it_is_invoked(self):
        cnn = get_cnn_model((31, 31, 9), n_outputs=16, convolutions=3, optimizer='adam', seed=26)
        ds_factory = RobotRebootDataSetFactory(31, cnn, max_depth=10, seed=26, playouts=1)
        v_1, p_1, s_1 = ds_factory.create(locate_robot_close_goal=False, max_movements=5)
        v_2, p_2, s_2 = ds_factory.create(locate_robot_close_goal=False, max_movements=5)
        self.assertFalse(np.array_equal(s_1, s_2), "States should be different everytime it is invoked")



