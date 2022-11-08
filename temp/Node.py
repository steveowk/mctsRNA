import math
import yaml
from random import sample, choice
from Utils import RNAUtils as RNAUtils

class TreeNode():
    def __init__(self, state, action_ix, parent = None, prior = 0.0):
        self.configs = yaml.load(open("Configs.yml", "r"), Loader = yaml.FullLoader)
        self.state = state
        self.parent = parent
        self.action_ix = action_ix
        self.children = {}
        self.visits = 0
        self.total_value = 0.
        self.prior = prior
        self.untried_actions = self.configs["paired"] if  self.state.paired(action_ix) else self.configs["unpaired"]
        #self.is_terminal = action_ix+1 >= self.state.max_seq_len
        self.is_terminal = self.state.is_terminal()
    
    def print_best_path(self):
        current = self
        while not current.is_terminal:
            print(current.state.designed)
            current = current.best_child(True)
        print(current.state)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def Q(self) -> float:
        return self.total_value / (1 + self.visits)

    def U(self) -> float:
        # return (math.sqrt(self.parent.visits)*self.prior / (1 + self.visits))
        return (math.sqrt(self.parent.visits) / (1 + self.visits))

    def best_child(self, infer = False):
        if infer : return max(self.children.values(), key = lambda node: node.Q())
        return max(self.children.values(), key = lambda node : node.Q() + node.U())

  
    def tree_policy(self):
        #print(f"tree policy start {self.state.designed}")
        current = self
        print(f"terminal {current.is_terminal} expanded {current.is_fully_expanded()} {current.children.keys()} {current.action_ix}")
        while not current.is_terminal:
            if not current.is_fully_expanded():
                print(f"expanding")
                return current.expand()
            else:
                current = current.best_child()
        print(f"tree policy end {current.state.designed}")
        return current

    def expand(self):
        action = self.untried_actions.pop()
        action_ix = self.action_ix
        new_state = self.state.copy_state()
        new_state.do_move(action, action_ix)
        self.children[action] = TreeNode(new_state, action_ix + 1, self)
        return self.children[action]

    def rollout(self):
        current = self.state.copy_state()
        print(f"rollout start {self.state.designed}")
        while self.action_ix < self.state.max_seq_len:
            action = choice(self.configs["paired"]) if current.paired(self.action_ix) else choice(self.configs["unpaired"])
            current.do_move(action, self.action_ix)
            self.action_ix += 1
        assert(current.is_terminal())
        print(f"rollout end {current.designed}")
        return current.value()

    def backup(self, v):
        current = self
        while current.parent is not None:
            current.visits += 1
            current.total_value += v
            current = current.parent