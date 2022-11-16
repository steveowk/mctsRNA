from mctsRNA.LoadRNA import LoadData as DataModel
import argparse
import yaml
import logging
from mctsRNA.RNAState import State as State
from mctsRNA.MCTS import MCTS as MCTS
from mctsRNA.DesignedRNA import Sequence as DesignedRNA
from copy import deepcopy
from joblib import Parallel, delayed

args = argparse.ArgumentParser()
args.add_argument('-r', '--root',  type=str, help="root dir", default="./")
args.add_argument('-t', '--test_mode',  type=int, help="test MCTS", default=1)
args.add_argument('-m', '--rolls', type=int, help="rollouts", default=1)
args.add_argument('-s', '--sims',  type=int,help=" MCTS simulations", default=300)
args.add_argument('-l', '--search', type=int,help="local search", default=0)
args.add_argument('-c', '--config', type=str, help="configs", default="./mctsRNA/config.yml")
args.add_argument('-v', '--verbose', type=bool, help="verbose", default=False)
args.add_argument('-x', '--mx_seq', type=int, help="max. seq.", default=200)
args.add_argument('-f', '--freq', type=int, help="print freq.", default=20)
args.add_argument('-d', '--dataset', type=str, help="dataset", default="modena")
args.add_argument('-p', '--workers', type=int, help="num processors", default=13)
args.add_argument('-e', '--interval_iter', type=int, help="sampling", default = 5)

args = args.parse_args()

# get the dataset for the train + the logs

config = yaml.load(open(args.config), Loader=yaml.FullLoader)
data_model = DataModel(**config['data'])
logging.log(logging.INFO, "Dataset loaded")
dataset = data_model.get_dataset(args.dataset)  
#if int(args.test_mode) ==1: dataset = dataset[:5]

def seq_processor(seq):
    seq_id = dataset.index(seq)
    if len(list(seq)) > args.mx_seq:return None
    state = State(seq)
    mcts = MCTS(state, args.sims, args.search,configs=config)
    designed_rna = DesignedRNA(state.target, config)
    action_ix = 0
    while not state.is_terminal():
        if action_ix%args.freq==0:
            print(f"current_seq {seq_id+1} action {action_ix+1} of {state.max_seq_len}")
        best_action = mcts.tree_search(action_ix=action_ix, verbose=args.verbose)
        paired, location = state.paired(action_ix, True)
        designed_rna.update(best_action, paired, location)
        state.designed = deepcopy(designed_rna.rna_seq)
        action_ix += 1
    assert(designed_rna.is_terminal() and  state.is_terminal() and mcts.state.is_terminal())
    designed_rna.write_results(seq_id+1, args.dataset)


def runner(iter):
    # the main loop
    
    Parallel(n_jobs=args.workers, verbose=args.freq)(delayed(seq_processor)(seq) for seq in dataset)
    filename = f"{config['result_path']}{args.dataset}/summary.csv"
    DesignedRNA.write_summary(filename, args.dataset, iter)
    
if __name__ == "__main__":

    # run the parallelised model to get samples of the size args.interval_iter

    for iter in range(args.interval_iter):runner(iter)
    DesignedRNA.generate_intervals()
    print("Done!!")