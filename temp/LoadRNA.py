from glob import glob

class LoadData():
    def __init__(self, **kwargs):
        self.runge_valid =  self.runge_modena_loader(kwargs["runge_valid_path"])
        self.modena = self.runge_modena_loader(kwargs["modena_path"])
        self.eterna = self.eterna_loader(kwargs["eterna_path"])
        self.runge_train = self.runge_modena_loader(kwargs["runge_train_path"])
        self.kauf_train = self.kauf_loader(kwargs["kauf_train_path"])
        self.kauf_valid = self.kauf_loader(kwargs["kauf_valid_path"])
    
    def runge_modena_loader(self, data_path) -> list:
        all_files = glob(data_path + "*.rna")
        data = []
        for f in all_files:
            with open(f, "r") as file:
                lines = file.readlines()
                assert(len(lines) == 1)
                lines = lines[0].strip()
                data.append(lines)
        return data

    def eterna_loader(self, data_path) -> list:
        with open(data_path, "r") as file:
            lines = file.readlines()
            data = [x.strip() for x in lines]
            assert(len(data) == 100)
        return data

    def kauf_loader(self, data_path) -> list:
        data = []
        with open(data_path, "r") as myfile:
            for line in myfile.readlines():
                line = line.strip()
                if line.startswith('.') or line.startswith('('):
                    data.append(line)
        return data