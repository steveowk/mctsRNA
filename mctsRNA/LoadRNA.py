from glob import glob

class LoadData:
    """
    This loads all the datasets for the model
     There are four datasets:
        1. Runge Valid - 63 sequences
        2. Modena - 29 sequences
        3. Eterna - 100 sequences
        4. Runge Train - 83 sequences
    """

    def __init__ (self, **kwargs):

        # setting up

        self.runge_valid =  self.runge_modena_loader(kwargs["runge_valid_path"])
        self.modena = self.runge_modena_loader(kwargs["modena_path"])
        self.eterna = self.eterna_loader(kwargs["eterna_path"])
        self.runge_train = self.runge_modena_loader(kwargs["runge_train_path"])
        self.kauf_train = self.kauf_loader(kwargs["kauf_train_path"])
        self.kauf_valid = self.kauf_loader(kwargs["kauf_valid_path"])
    
    # get data funtion helper 

    def get_dataset(self, name):
        if name == "runge_valid":
            return self.runge_valid
        elif name == "modena":
            return self.modena
        elif name == "eterna":
            return self.eterna
        elif name == "runge_train":
            return self.runge_train
        elif name == "kauf_train":
            return self.kauf_train
        elif name == "kauf_valid":
            return self.kauf_valid
        else:
            raise ValueError("Invalid data name")

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