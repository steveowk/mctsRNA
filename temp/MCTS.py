from Node import TreeNode as TreeNode
from RNAState import State as State

class MCTS():
    def __init__(self, state, num_simulations, time_in_seconds = None) -> None:
        self.root_state = state
        self.num_simulations = num_simulations
    def search(self):
        root = TreeNode(self.root_state, 0)
        for _ in range(self.num_simulations):
            leaf = root.tree_policy()
            v = leaf.rollout()
            leaf.backup(v)
        print(f"designed {leaf.state.designed} value {v}")

test = "..().."
state = State(test)
mcts = MCTS(state, 10)
mcts.search()