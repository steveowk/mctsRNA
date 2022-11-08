from mctsRNA.LoadRNA import LoadData as Data
import argparse
import yaml
from mctsRNA.RNAState import State as State
from mctsRNA.MCTS import MCTS as MCTS
from mctsRNA.DesignedRNA import Sequence as DesignedRNA
from copy import deepcopy

args = argparse.ArgumentParser()
args.add_argument('-r', '--root',  type=str, help="root dir", default="./")
args.add_argument('-t', '--test',  type=bool, help="test MCTS", default=False)
args.add_argument('-m', '--rolls', type=int, help="rollouts", default=1)
args.add_argument('-s', '--sims',  type=int,help=" MCTS simulations", default=100)
args.add_argument('-l', '--search', type=bool,help="local search", default=True)
args.add_argument('-c', '--config', type=str, help="configs", default="./mctsRNA/config.yml")
args.add_argument('-v', '--verbose', type=bool, help="verbose", default=False)
args.add_argument('-x', '--mx_seq', type=int, help="max. seq.", default=200)
args = args.parse_args()

def main(verbose=False):
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    data_loader = Data(**config["data"])
    datasets = [data_loader.eterna, data_loader.runge_valid,
                data_loader.modena,data_loader.kauf_train,
                data_loader.kauf_valid]
    data_name = {0: "eterna", 1: "runge_valid",
                 2: "modena", 2: "kauf_train", 4: "kauf_valid"}
    if args.test: datasets = [[x[0]] for x in datasets]
    for data_id, data in enumerate(datasets):  # for each dataset
        for seq_id, seq in enumerate(data):  # for each sequence
            if len(list(seq)) > args.mx_seq:continue    
            state = State(seq)
            mcts = MCTS(state, args.sims, args.search,configs=config)
            designed_rna = DesignedRNA(state.target, config)
            action_ix = 0
            while not state.is_terminal():
                if verbose:print(f"{action_ix} len {len(list(seq))}")
                best_action = mcts.tree_search(action_ix=action_ix, verbose=args.verbose)
                paired, location = state.paired(action_ix, True)
                designed_rna.update(best_action, paired, location)
                state.designed = deepcopy(designed_rna.rna_seq)
                action_ix += 1
            assert(designed_rna.is_terminal() and state.is_terminal()
                    and mcts.state.is_terminal())
            designed_rna.write_results(seq_id+1, data_name[data_id])
            print(f"Finished {seq_id+1} of {len(data)} in {data_id+1} of {len(datasets)}")

if __name__ == "__main__":
    main()
    print("Done!!")
