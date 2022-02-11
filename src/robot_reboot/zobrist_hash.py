from abc import ABC, abstractmethod


class ZobristHash(ABC):

    @abstractmethod
    def robots_count(self):
        pass

    @abstractmethod
    def maze_shape(self):
        pass

    @abstractmethod
    def get_value(self):
        pass

    @abstractmethod
    def empty(self):
        pass
