import math
from mctsRNA.Node import TreeNode as TreeNode


class MCTS():
    def __init__(self, state, simulations, search, configs):
        self.state = state
        self.simulations = simulations
        self.search = search
        self.configs = configs
    def tree_search(self, action_ix, verbose=False):
        root = TreeNode(self.state, action_ix, configs=self.configs) # root node
        for _ in range(self.simulations):
            leaf = root.select_leaf() # select and expand node 
            v = leaf.rollout(search=self.search, verbose=verbose)  #rollout
            leaf.back_up(v) # backpropagate the ac. value
        if verbose:
            for key, node in root.children.items():
                print(f" {key} -> {node.U()} visits {node.visits} value {node.Q()}")
        best_action = root.best_child(infer=True).to_here
        if verbose:print(f"best action {best_action}")
        return best_action