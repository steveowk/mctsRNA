import math
import yaml
from random import choice, shuffle
from copy import deepcopy


class TreeNode():
    def __init__(self, state, action_ix, to_here=None, parent=None, prior=False, configs=None):
        self.state = state
        self.parent = parent
        self.action_ix = action_ix
        self.children = {}
        self.visits = 1
        self.total_value = 0.
        self.to_here = to_here
        self.configs = configs
        self.prior = self.get_prior(to_here) if prior else 1
        shuffle(self.configs["paired"])
        shuffle(self.configs["unpaired"])
        self.C = math.sqrt(2)
        self.untried_actions = deepcopy(self.configs["paired"]) if state.paired(
            action_ix) else deepcopy(self.configs["unpaired"])
        shuffle(self.untried_actions)
        

    def fully_expanded(self):
        return len(self.untried_actions) == 0

    def get_prior(self, to_here):  # prior base distribution
        prior = self.configs["paired_prob"][to_here] if self.to_here in self.configs[
            "paired"] else self.configs["unpaired_prob"][to_here]
        return prior

    def Q(self) -> float:  # q function
        return self.total_value / self.visits

    def U(self) -> float:  # explore
        return self.C * self.prior * \
            math.sqrt(2.0 * math.log(self.parent.visits) / self.visits)

    def best_child(self, infer=False):  # UCB formula
        if infer:
            # inference
            return max(self.children.values(), key=lambda node: node.Q())
        else:
            return max(self.children.values(), key=lambda node: node.Q() + node.U())

    def select_leaf(self):
        current = self
        while not current.state.is_terminal():
            if not current.fully_expanded():
                return current.expand_leaf()
            else:
                current = current.best_child()
        return current

    def expand_leaf(self):  # expand leaf node
        action_ix = self.action_ix
        action = self.untried_actions.pop()
        new_state = self.state.copy_state()
        new_state.do_move(action, action_ix)
        if not new_state.is_terminal():action_ix += 1
        self.children[action] = TreeNode(state=new_state, action_ix=action_ix, 
                                to_here=action, parent=self, prior=True, configs=self.configs)
        return self.children[action]

    def rollout(self, search=False, verbose=False):  # simulation phase
        current = self.state.copy_state()
        action_ix = self.action_ix
        while action_ix < self.state.max_seq_len and not current.is_terminal():
            action = choice(self.configs["paired"]) if current.paired(
                action_ix) else choice(self.configs["unpaired"])
            current.do_move(action, action_ix)
            action_ix += 1
        assert (current.is_terminal())
        v = int(current.value() * 100)
        hamming_score = 100 - v
        mx_threshold = self.configs["mutation_threshold"]
        improve = v < 100 and hamming_score <= mx_threshold
        if verbose:
            print(
                f" value -> {v} hamming -> {hamming_score} improve -> {improve}")
        if search and improve:
            current.local_search(self.configs["max_sample"], mx_threshold)
            if verbose:
                print(f"Before LS {v/100} : After LS {current.value()}")
        return current.value()

    def back_up(self, result):  # backpropagation of value [ accuracy]
        self.visits += 1.
        self.total_value += result
        if self.parent:
            self.parent.back_up(result)
