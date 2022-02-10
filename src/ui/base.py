from abc import ABC, abstractmethod


class GameStateView(ABC):
    def __init__(self, agent):
        self._agent = agent

    @property
    def agent(self):
        return self._agent

    @abstractmethod
    def update(self):
        pass
