import RNA
import csv
from sklearn.metrics import accuracy_score
import os
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
    def write_results(self, seq_id, data_id):
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
        with open(f"{self.config['result_path']}/{data_id}/summary.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(row)
