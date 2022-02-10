from abc import ABC


class Encoder(ABC):
    def name(self):
        raise NotImplementedError()

    def encode(self, game_state):
        raise NotImplementedError()

    def encode_action(self, action):
        raise NotImplementedError()

    def decode_action_index(self, index):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()
