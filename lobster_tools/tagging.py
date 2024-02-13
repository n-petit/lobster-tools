import datetime as dt
import itertools as it
import pickle
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from lobster_tools.flow import _FEATURE_NAMES_WITH_MARGINALS, evaluate_features
from lobster_tools.preprocessing import Event

flags.DEFINE_multi_float(
    "quantiles",
    [0.1, 0.9],
    "Quantiles for the features.",
    lower_bound=0.0,
    upper_bound=1.0,
)

flags.DEFINE_integer(
    "lookback",
    3,
    "Lookback period in days for rolling quantiles.",
)

FLAGS = flags.FLAGS

_OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

def main(_):
    # load data
    tolerance = FLAGS.tolerances[0]  # single tolerance for now

    etf_dir = _OUTPUT_DIR / "latest" / FLAGS.etf
    tol_dir = etf_dir / tolerance

    etf_executions = pd.read_pickle(etf_dir / "etf_executions.pkl")

    features = pd.read_pickle(tol_dir / "features.pkl")
    neighbors = pd.read_pickle(tol_dir / "neighbors.pkl")

    # compute rolling quantiles
    quantiles = features.groupby(features.index.date).quantile(FLAGS.quantiles)
    quantiles.index = quantiles.index.set_names(["date", "quantile"])

    rolling_quantiles = (
        quantiles.groupby(level="quantile").rolling(FLAGS.lookback).mean()
    )
    assert rolling_quantiles.index.names == ["quantile", "date", "quantile"]
    rolling_quantiles.index = rolling_quantiles.index.droplevel(-1)
    rolling_quantiles = rolling_quantiles.dropna()
    assert rolling_quantiles.ge(0).all().all()

    # restrict
    tmp = rolling_quantiles.index.get_level_values("date")
    first_date = tmp.unique().asof(tmp.min() + dt.timedelta(days=1))
    features = features[features.index.date >= first_date]

    # tags
    tags = pd.DataFrame(index=features.index, columns=features.columns)

    for date, group in features.groupby(features.index.date):
        previous_trading_day = (
            rolling_quantiles.index.get_level_values("date")
            .unique()
            .asof(date - dt.timedelta(days=1))
        )
        quantiles_for_date = rolling_quantiles.xs(previous_trading_day, level="date")

        for column in group.columns:
            bins = quantiles_for_date[column].values

            # TODO: this is not good. fix this. i.e ETF_proportion or whatever
            # bins may have different lengths on each iteration
            bins = np.append(bins, np.inf)
            if bins[0] != 0.0:
                bins = np.insert(bins, 0, 0.0)
            bins = np.unique(bins)

            tags.loc[group.index, column] = pd.cut(
                group[column],
                bins=bins,
                labels=range(len(bins) - 1),
                include_lowest=True,
            )

    full = pd.concat((etf_executions.drop(columns="notional"), neighbors), axis=1).join(
        tags, how="left"
    )
    full.loc[~full.nonIso, _FEATURE_NAMES_WITH_MARGINALS] = -1

    full = full.astype({col: 'category' for col in _FEATURE_NAMES_WITH_MARGINALS})

    # save to disk



if __name__ == "__main__":
    app.run(main)
