import os
import sys
sys.path.append("../")
from sklearn.metrics import accuracy_score
import RNA
import math
import RNAState
from random import sample


seq = ".(..)."  # 0,(1,4),2,3,5
# target CUGUGC, available mov positions 0,(1,4), 2, 3, 5 -> {0 : (1, 4),
# 1 : 0, 2: 2, 3 : 3, 4 : 5}
state = RNAState.State(seq)


def action_base(ix):
    ix_decode = state.paired_unpaired_ix[ix]
    if ix_decode in state.paired_locations:
        return "GC", ix
    return "U", ix


for i in range(state.max_seq_len):
    state.do_move(*action_base(i))
print(f"state designed {state.designed} accuracy {state.value()} max_seq_len {state.max_seq_len} {state.target_RNA() == seq } {state.designed_RNA()=='UGUUCU'}")

ac = accuracy_score(list(seq), list(RNA.fold(state.designed_RNA())[0]))

if state.max_seq_len == 5 and state.target_RNA() == seq and state.designed_RNA() == "UGUUCU": #and math.isclose(state.value() - 0.6666666666, 0.0):
    print(f"All tests passed!")
else:
    print(f"Error: tests failed")
