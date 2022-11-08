from copy import deepcopy
from sklearn.metrics import accuracy_score
import RNA
from random import shuffle
from itertools import product
from sklearn.metrics import accuracy_score
from random import sample


class State():
    def __init__(self, rna_seq: str) -> None:
        self.target = list(rna_seq)
        self.designed = [None] * len(self.target)
        self.paired_locations, self.unpaired_locations = self.get_parings(rna_seq)
        self.max_seq_len = len(self.paired_locations) + len(self.unpaired_locations)
        self.valid_actions = self.paired_locations + self.unpaired_locations
        shuffle(self.valid_actions)
        self.paired_unpaired_ix = {ix: loc for ix, loc in enumerate(self.valid_actions)}

    def set_designed(self, designed):
        self.designed = designed

    def value(self, designed=None):
        if designed is None:
            assert (None not in self.designed)
            assert(len(self.designed) == len(self.target))
            ac = accuracy_score(self.target, list(
                RNA.fold("".join(self.designed))[0]))
            return ac
        else:
            assert (isinstance( designed,str))
            assert(len(designed) == len(self.target))
            ac = accuracy_score(self.target, list(
                RNA.fold("".join(designed))[0]))
            return ac

    def paired(self, action_ix, return_pair=False):
        action_decode = self.paired_unpaired_ix[action_ix]
        paired = None
        if action_decode in self.unpaired_locations:
            paired = False
        else:
            paired = True
        if not return_pair:
            return paired
        else:
            return paired, action_decode

    def copy_state(self):
        return deepcopy(self)

    def do_move(self, action: str, action_ix: int):  # execute the move
        if action_ix >= self.max_seq_len or None not in self.designed:
            return
        if not self.paired(action_ix):
            assert (len(list(action)) == 1)
            loc_x = self.paired_unpaired_ix[action_ix]
            if self.designed[loc_x] is None:
                self.designed[loc_x] = action
            else:
                print("single already assigned")
        else:
            assert (len(list(action)) == 2)
            loc_x, loc_y = self.paired_unpaired_ix[action_ix]
            if self.designed[loc_x] is None and self.designed[loc_y] is None:
                self.designed[loc_x] = action[0]
                self.designed[loc_y] = action[1]
            else:
                print("pair already assigned")

    def designed_RNA(self):
        return "".join(self.designed)

    def target_RNA(self):
        return "".join(self.target)

    def is_terminal(self):
        if None in self.designed:return False
        return True

    def seq_mutation(self, mutation, sites):
        mutated_seq = deepcopy(self.designed)
        for site, mutation in zip(sites, mutation):
            mutated_seq[site] = mutation
        return mutated_seq

    def scaled_accuracy(self, seq):
        assert (len(seq) == len(self.target))
        assert (isinstance(seq, list))
        return int(accuracy_score(seq, self.target) * len(seq))

    def local_search(self, max_samples, max_diff, verbose=False):
        best_score = int(self.value()*100)
        ds = list(RNA.fold(self.designed_RNA())[0])
        ts = deepcopy(self.target)
        mutating_sites = [ix for ix in range(len(ds)) if ds[ix] != ts[ix]]
        if verbose:print(f"mutation size -> {len(mutating_sites)}")
        mutated_paths = list(product("AGCU", repeat=min(max_diff, len(mutating_sites))))
        shuffle(mutated_paths)
        mutated_paths = sample(mutated_paths, max_samples) if len(
            mutated_paths) > max_samples else mutated_paths
        for mutation in mutated_paths:
            mutated = self.seq_mutation(mutation, mutating_sites)
            folded_mutated = list(RNA.fold("".join(mutated))[0])
            v = self.scaled_accuracy(folded_mutated)
            if v == 100:
                self.designed = mutated
                return
            if v > best_score:
                self.designed = mutated
                best_score = v

    def get_parings(self, target):
        stack = list()
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
