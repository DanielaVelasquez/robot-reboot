class NeuralNetworkOutputNotMatchingGameActions(Exception):
    def __init__(self):
        self.message = "Number of outputs for p int the neural network don't match the number of actions in the game"
