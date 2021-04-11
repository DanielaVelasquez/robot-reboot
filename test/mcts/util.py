import numpy as np


def assert_state(mcts, s, n=[], w=[], p=[], message=""):
    np.testing.assert_equal(mcts.states_statistics[s].n, n, message)
    np.testing.assert_equal(mcts.states_statistics[s].w, w, message)
    np.testing.assert_equal(mcts.states_statistics[s].p, p, message)
