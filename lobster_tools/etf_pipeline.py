# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/08_etf_pipeline.ipynb.

# %% auto 0
__all__ = ['cfg']

# %% ../notebooks/08_etf_pipeline.ipynb 5
import os

import click
from arcticdb import Arctic, LibraryOptions
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
from pathlib import Path
from .config import MainConfig, register_configs
from .preprocessing import *
from .querying import *
from .flow_decomposition import *
import pandas as pd
from sklearn.linear_model import LinearRegression
from itertools import product
import datetime
from dataclasses import dataclass
from functools import partial
import json
import numpy as np
from pprint import pprint

# %% ../notebooks/08_etf_pipeline.ipynb 6
# access config by normal python import
cfg = MainConfig()
# register configs and then build object
register_configs()
with initialize(version_base=None, config_path=None):
    cfg_omega = compose(config_name="config")
    cfg = OmegaConf.to_object(compose(config_name="config"))
    print(cfg)
    print(cfg.universe.equities)
