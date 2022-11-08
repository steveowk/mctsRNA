import yaml
from LoadRNA import LoadData as RNAData
from random import sample
config = yaml.load(open("Configs.yml", "r"), Loader = yaml.FullLoader)

data = RNAData(**config["data"])

print(f"runge_train {len(data.runge_train)}, runge_valid {len(data.runge_valid)}\
    , kauf_train {len(data.kauf_train)}, kauf_valid {len(data.kauf_valid)}\
    , eterna {len(data.eterna)}, modena {len(data.modena)}")
print(f"runge_train {sample(data.runge_train, 1)}, runge_valid {sample(data.runge_valid, 1)}\
    , kauf_train {sample(data.kauf_train, 1)}, kauf_valid {sample(data.kauf_valid, 1)}\
    , eterna {sample(data.eterna, 1)}, modena {sample(data.modena, 1)}")