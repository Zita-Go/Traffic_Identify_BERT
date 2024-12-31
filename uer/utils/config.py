import json
from argparse import Namespace


def load_hyperparam(args):
    # Load hyperparameters from config files for encoder1 and encoder2.
    with open(args.config_path1, mode="r", encoding="utf-8") as f:
        param1 = json.load(f)
    with open(args.config_path2, mode="r", encoding="utf-8") as f:
        param2 = json.load(f)
    
    param = {}
    for key in set(param1.keys()).union(param2.keys()):
        param[key] = {}
        if key in param1:
            param[key]["encoder1"] = param1[key]
        if key in param2:
            param[key]["encoder2"] = param2[key]

    args_dict = vars(args)
    args_dict.update(param)
    args = Namespace(**args_dict)

    return args
