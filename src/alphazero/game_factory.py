from abc import ABC, abstractmethod


class GameFactory(ABC):
    def __init__(self, seed):
        self.seed = seed

    @abstractmethod
    def build(self):
        """Builds a game"""
        pass

    @abstractmethod
    def build_with(self, seed):
        """Builds a game based on a seed value, results are reproducible over time"""
        pass
