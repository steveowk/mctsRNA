from copy import deepcopy
from sklearn.metrics import accuracy_score
import RNA


class State():
    def __init__(self, rna_seq: str) -> None:
        self.target = list(rna_seq)
        self.designed = [None] * len(self.target)
        self.paired_locations, self.unpaired_locations = self.paired_and_unpaired(rna_seq)
        self.max_seq_len = len(self.paired_locations) +  len(self.unpaired_locations)
        self.paired_unpaired_ix = {ix: baseloc for ix, baseloc in enumerate(self.paired_locations + self.unpaired_locations)}

    def value(self, designed=None) -> float:
        if designed is None:
            assert (None not in self.designed and len(
                self.designed) == len(self.target))
            return accuracy_score(self.target, list(RNA.fold("".join(self.designed))[0]))
        else:
            assert (isinstance(designed, str) and len(designed) == len(self.target))
            return accuracy_score(self.target, list(RNA.fold("".join(designed))[0]))
    
    def paired(self, action_ix) -> bool:
        action_decode = self.paired_unpaired_ix[action_ix]
        if action_decode in self.unpaired_locations:return False
        return True

    def copy_state(self):
        state = State(self.target)
        state.designed = deepcopy(self.designed)
        return state

    def do_move(self, action: str, action_ix: int) -> None:
        #print(f"action: {action}, action_ix: {action_ix}, paired: {self.paired(action_ix)} {list(action)}")
        if action_ix >= self.max_seq_len or None not in self.designed:
            print("Error: action_ix is greater than max_seq_len or designed is full.")
            return
        if not self.paired(action_ix):
            assert (len(list(action)) == 1)
            loc_x = self.paired_unpaired_ix[action_ix]
            self.designed[loc_x] = action
        else:
            assert (len(list(action)) == 2)
            loc_x, loc_y = self.paired_unpaired_ix[action_ix]
            if self.designed[loc_x] is None and self.designed[loc_y] is None:
                self.designed[loc_x] = action[0]
                self.designed[loc_y] = action[1]

    def paired_and_unpaired(self, target) -> tuple:
        stack = []
        paired_bases = list()
        unpaired_bases = list()
        for i in range(len(target)):
            if target[i] == '(':
                stack.append(i)
            if target[i] == ')':
                paired_bases.append((stack.pop(), i))
            elif target[i] == '.':
                unpaired_bases.append(i)
        del stack
        return (paired_bases, unpaired_bases)

    def designed_RNA(self): return "".join(self.designed)
    def target_RNA(self): return "".join(self.target)
    def is_terminal(self):
        if None in self.designed: return False
        return True
