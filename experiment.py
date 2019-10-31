from sacred import Experiment
from sacred.observers import MongoObserver
import toml
import numpy as np
import torch

ex = Experiment('example')
ex.add_config(toml.load('config.toml'))
ex.observers.append(MongoObserver())

@ex.automain
def main(_config, _rnd, _run):
    _rnd = np.random.RandomState(seed=_config['model_params']['seed'])
    torch.random.manual_seed(seed=_config['model_params']['seed'])
