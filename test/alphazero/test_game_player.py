import unittest
from unittest.mock import Mock

from exceptions.exceptions import RequiredValueException
from src.alphazero.game_player import GamePlayer
from test.alphazero.fake_data import fn_predict_probability_1_for_next_action, FakeModel, FakeGame, FakeState


def get_mock_model():
    mock_model = Mock()
    mock_model.predict.return_value = [0.45, 0.11, 0.26, 0.18], 1
    return mock_model


class TestGamePlayer(unittest.TestCase):
    def test_init(self):
        mock_model = get_mock_model()
        mock_game = Mock()
        player = GamePlayer(mock_model, mock_game)
        self.assertEqual(mock_model, player.model)
        self.assertEqual(mock_game, player.game)

    def test_fails_when_model_is_none(self):
        self.assertRaises(RequiredValueException, lambda: GamePlayer(None, Mock()))

    def test_fails_when_game_is_none(self):
        self.assertRaises(RequiredValueException, lambda: GamePlayer(get_mock_model(), None))

    def test_predict_uses_model(self):
        player = GamePlayer(get_mock_model(), Mock())
        self.assertEqual(([0.45, 0.11, 0.26, 0.18], 1), player.predict(Mock()))

    def test_predict_fails_when_state_is_none(self):
        player = GamePlayer(get_mock_model(), Mock())
        self.assertRaises(RequiredValueException, lambda: player.predict(None))

    def test_play(self):
        fake_game = FakeGame()
        fake_model = FakeModel(fn_predict_probability_1_for_next_action, fake_game)
        fake_state = FakeState(fake_game, 0)
        player = GamePlayer(fake_model, fake_game)
        final_state = player.play(fake_state)
        self.assertEqual(3, final_state.value)
        self.assertEqual(3, final_state.sequence_i)

    def test_play_when_max_actions_reached(self):
        fake_game = FakeGame()
        fake_model = FakeModel(fn_predict_probability_1_for_next_action, fake_game)
        fake_state = FakeState(fake_game, 0)
        player = GamePlayer(fake_model, fake_game)
        final_state = player.play(fake_state, max_actions=2)
        self.assertEqual(2, final_state.value)
        self.assertEqual(2, final_state.sequence_i)
