class Edge:
    def __init__(self):
        self.__n = 0  # Number of times action has been taken from state S
        self.__w = 0  # Total value of the next state
        self.__q = 0  # Mean value of next state
        self.__p = 0  # Prior probability to select current action

    @property
    def n(self):
        return self.__n

    @property
    def w(self):
        return self.__w

    @property
    def q(self):
        return self.__q

    @property
    def p(self):
        return self.__p

    def count_visit(self):
        self.__n += 1

    def update_probability(self, p):
        if 0 > p > 1:
            raise Exception("Invalid probability value")
        self.__p = p

    def update_next_state_value(self, v):
        if self.__n == 0:
            raise Exception("No visits have been registered")
        self.__w += v
        self.__q = self.__w / self.__n
