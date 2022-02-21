import numpy as np

__all__ = [
    'AlphaZeroAgent',
]

from keras.optimizer_v2.gradient_descent import SGD

from src.agent.base import Agent


class Branch:
    def __init__(self, prior, next_state):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.next_state = next_state


class AlphaZeroTreeNode:
    def __init__(self, state, value, priors, parent, last_action):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_action = last_action
        self.total_visit_count = 1
        self.branches = {}
        valid_actions_to_state_map = state.get_valid_actions_next_state_map()
        for action in valid_actions_to_state_map:
            p = priors[action]
            next_state = valid_actions_to_state_map[action]
            self.branches[action] = Branch(p, next_state)
        self.children = {}

    def actions(self):
        return self.branches.keys()

    def add_child(self, action, child_node):
        self.children[action] = child_node

    def has_child(self, action):
        return action in self.children

    def get_child(self, action):
        return self.children[action]

    def record_visit(self, action, value):
        self.total_visit_count += 1
        self.branches[action].visit_count += 1
        self.branches[action].total_value += value

    def expected_value(self, action):
        branch = self.branches[action]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, action):
        return self.branches[action].prior

    def next_state(self, action):
        return self.branches[action].next_state

    def visit_count(self, action):
        if action in self.branches:
            return self.branches[action].visit_count
        return 0


class AlphaZeroAgent(Agent):

    def __init__(self, model, encoder=None, rounds_per_action=1600, c=2.0, collector=None):
        self.model = model
        self.encoder = encoder

        self.collector = collector

        self.num_rounds = rounds_per_action
        self.c = c

    def select_action(self, game_state):

        root = self.__create_node(game_state)

        for i in range(self.num_rounds):
            node = root
            next_action = self.__select_branch(node)
            while node.has_child(next_action):
                node = node.get_child(next_action)
                next_action = self.__select_branch(node)

            new_state = node.next_state(next_action)
            child_node = self.__create_node(
                new_state, action=next_action, parent=node)

            action = next_action
            value = child_node.value
            while node is not None:
                node.record_visit(action, value)
                action = node.last_action
                node = node.parent
                # value = -1 * value

        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array([
                root.visit_count(
                    self.encoder.decode_action_index(idx))
                for idx in range(self.encoder.num_actions())
            ])
            self.collector.record_decision(
                root_state_tensor, visit_counts)

        return max(root.actions(), key=root.visit_count)

    def __select_branch(self, node):
        total_n = node.total_visit_count

        def score_branch(action):
            q = node.expected_value(action)
            p = node.prior(action)
            n = node.visit_count(action)
            return q + self.c * p * np.sqrt(total_n) / (n + 1)

        return max(node.actions(), key=score_branch)

    def __create_node(self, game_state, action=None, parent=None):
        state_tensor = self.encoder.encode(game_state)
        model_input = np.array([state_tensor])
        priors, values = self.model.predict(model_input)
        priors = priors[0]
        value = values[0][0]
        action_priors = {
            self.encoder.decode_action_index(idx): p
            for idx, p in enumerate(priors)
        }
        new_node = AlphaZeroTreeNode(
            game_state, value,
            action_priors,
            parent, action)
        if parent is not None:
            parent.add_child(action, new_node)
        return new_node

    def train(self, experience, learning_rate, batch_size):
        num_examples = experience.states.shape[0]

        model_input = experience.states

        visit_sums = np.sum(
            experience.visit_counts, axis=1).reshape(
            (num_examples, 1))
        action_target = experience.visit_counts / visit_sums

        value_target = experience.rewards

        self.model.compile(
            SGD(learning_rate=learning_rate),
            loss=['categorical_crossentropy', 'mse'])
        self.model.fit(
            model_input, [action_target, value_target],
            batch_size=batch_size)
