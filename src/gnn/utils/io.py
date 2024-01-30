import json
from argparse import Namespace
from collections import OrderedDict, namedtuple


def load_config(path, as_dict=True):
    with open(path, "r") as f:
        config = json.load(f)
        if not as_dict:
            config = namedtuple("config", config.keys())(*config.values())
    return config


def save_config(config, path):
    if isinstance(config, dict):
        pass
    elif isinstance(config, Namespace):
        config = vars(config)
    else:
        try:
            config = config._as_dict()
        except BaseException as e:
            raise e

    with open(path, "w") as f:
        json.dump(config, f)
