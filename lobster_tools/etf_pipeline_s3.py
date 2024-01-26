__all__ = ["quick_marginalize"]

import datetime as dt
import itertools as it
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arcticdb import Arctic, QueryBuilder
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from lobster_tools.config import (
    NASDAQExchange,
    Overrides,
    etf_to_equities,
    get_config,
)
from lobster_tools.flow_decomposition import *
from lobster_tools.preprocessing import *


def quick_marginalize(df: pd.DataFrame) -> pd.DataFrame:
    """Manual implementation but for all features."""
    df = df.copy()
    # NOTE: tmp hardcoded feature names, change this to a regexp
    # will just grab these from globals and not set in function.
    tolerances = ["250us", "500us"]
    features = ["num_trades", "notional", "distinct_tickers"]
    col_names = ["_" + "_".join(x) for x in it.product(tolerances, features)]

    for col_name in col_names:
        # marginalise over all
        df[col_name] = (
            df[col_name + "_ss_af"]
            + df[col_name + "_ss_bf"]
            + df[col_name + "_os_af"]
            + df[col_name + "_os_bf"]
        )
        # marginalise over bf/af
        df[col_name + "_ss"] = df[col_name + "_ss_af"] + df[col_name + "_ss_bf"]
        df[col_name + "_os"] = df[col_name + "_os_af"] + df[col_name + "_os_bf"]
        # marginalise over ss/os
        df[col_name + "_af"] = df[col_name + "_ss_af"] + df[col_name + "_os_af"]
        df[col_name + "_bf"] = df[col_name + "_ss_bf"] + df[col_name + "_os_bf"]

    return df
