import RNA
import csv
import pandas as pd
from sklearn.metrics import accuracy_score
import os
import numpy as np
from glob import glob
from os import path


class Sequence():
    def __init__(self, target, config):
        self.target = target
        self.config = config
        self.rna_seq = [None] * len(target)
    def update(self, best, paired, location):
        if paired:
            assert (len(list(best)) == 2)
            self.rna_seq[location[0]] = best[0]
            self.rna_seq[location[1]] = best[1]
        else:
            assert (len(list(best)) == 1)
            self.rna_seq[location] = best
    def is_terminal(self):
        return None not in self.rna_seq
    
    """
    this function writes the results of each sequence to a file
    """
    def write_results(self, seq_id, data_id, _iter):
        """
        Function to write the result of the sample to a file

        """
        assert (self.is_terminal())
        if not path.exists(self.config["result_path"]):
            os.mkdir(self.config["result_path"])
        if not path.exists(f"{self.config['result_path']}/{data_id}"):
            os.mkdir(f"{self.config['result_path']}/{data_id}")
        seq = "".join(self.rna_seq)
        seq_fold = list(RNA.fold(seq)[0])
        assert (len(seq_fold) == len(seq))
        ac = accuracy_score(self.target, seq_fold)
        row = [seq_id, int(ac * 100), "".join(self.rna_seq),"".join(self.target)]
        with open(f"{self.config['result_path']}/{data_id}/summary-{_iter}-.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    """
    Function to calculate the stats of the results at the end of the sample run
    """

    @classmethod
    def write_summary(cls,  filename, desc, iter_id):
        # data saved

        data = pd.read_csv(filename, names=["seq_id", "accuracy", "seq", "target"])
        
        # get the average accuracies and compute GC and the energy of the sequences
        
        data['energy'] = data.seq.apply(lambda x: RNA.fold(x)[-1])
        data['gc'] = data.seq.apply(lambda x: round((list(x).count('C') + 
                                    list(x).count('G')) / len(list(x)) * 100))
        # some stats we are interested in

        total = data.shape[0]                           
        accuracy100 = data.loc[data['accuracy'] == 100]['accuracy'].shape[0] # acc 
        energy100 = data.loc[data['accuracy'] == 100].energy.mean()
        gc_score =  data.loc[data['accuracy'] == 100].gc.mean()
        above90 =   data.loc[(data['accuracy'] >=90) & (data['accuracy'] < 100)]['accuracy'].shape[0] # acc above 90
        energy90 =  data.loc[(data['accuracy'] >=90) & (data['accuracy'] < 100)].energy.mean()
        gc_score90 = data.loc[(data['accuracy'] >=90) & (data['accuracy'] < 100)].gc.mean()
        with open(f"./results/All_results-{iter_id}-.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([desc, total, f"{accuracy100:.2f}", f"{energy100:.2f}", f"{gc_score:.2f}", 
                            f"{above90:.2f}", f"{energy90:.2f}", f"{gc_score90:.2f}"])  
    @classmethod
    def generate_intervals(cls):
        summaries = glob("./results/All_results-*.csv")
        names = ["model", "total", "accuracy100", "energy100", "gc_score", "above90", "energy90", "gc_score90"]
        df = [pd.read_csv(d,names=names) for d in summaries]
        final = pd.DataFrame([]) # no names,  thank me later
        for d in df:final = pd.concat([d,final])
        const = 1.96
        cols = list(final.columns)
        cols.remove("model")
        cols.remove("total")
        ixs = set(final.model.values)
        ixs = [x.strip() for x in ixs]
        stats = final.groupby(['model'])["accuracy100", "energy100", "gc_score","above90", "energy90", "gc_score90"].agg(['sem', 'mean'])
        for ix in ixs:
            for col in cols:
                sem = stats.loc[ix][col]['sem'] * const
                stats.loc[ix][col]['sem'] = sem
        stats.to_csv("./results/intervals.csv", index=True)