from abc import ABC, abstractmethod


class Encoder(ABC):
    @abstractmethod
    def name(self):
        raise NotImplementedError()

    @abstractmethod
    def encode(self, game_state):
        raise NotImplementedError()

    @abstractmethod
    def encode_action(self, action):
        raise NotImplementedError()

    @abstractmethod
    def decode_action_index(self, index):
        raise NotImplementedError()

    @abstractmethod
    def shape(self):
        raise NotImplementedError()

    @abstractmethod
    def num_actions(self):
        raise NotImplementedError()
