from absl import flags
from absl import logging
from absl import app

import datetime as dt
import pandas as pd
import numpy as np
import typing as t
import itertools as it

from arcticdb import QueryBuilder
from pathlib import Path

from lobster_tools.read_data_from_s3 import fetch_tickers, assign_mid
from lobster_tools.preprocessing import Event, EventGroup
from arcticdb import Arctic

from numpy.typing import NDArray
from sklearn.neighbors import KDTree

import pickle

flags.DEFINE_list(
    "tolerances",
    ["20ms"],
    "Epsilon neighbors in time.",
)
flags.DEFINE_string(
    "etf",
    "XLC",
    "ETF to run pipeline on.",
)

FLAGS = flags.FLAGS


def get_times(df: pd.DataFrame) -> NDArray[np.datetime64]:
    "Return numpy array of times from the index of the DataFrame."
    if df.index.values.dtype != "datetime64[ns]":
        raise TypeError("DataFrame index must be of type datetime64[ns]")
    return df.index.values.reshape(-1, 1)


def str_to_time(time: str, convert_to: str) -> int:
    return pd.Timedelta(time) / pd.Timedelta(1, unit=convert_to)


def str_to_nanoseconds(time: str) -> int:
    return int(str_to_time(time, convert_to="ns"))


def aggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    duplicates = df[df.duplicated(subset=["datetime", "direction"], keep=False)]

    aggregated_sizes = duplicates.groupby(["datetime", "direction"])["size"].transform(
        "sum"
    )

    df.loc[duplicates.index, "size"] = aggregated_sizes
    _MERGED_ORDER_ID = -2
    df.loc[duplicates.index, ["event", "order_id"]] = [
        Event.AGGREGATED.value,
        _MERGED_ORDER_ID,
    ]
    df = df.drop_duplicates(subset=["datetime"], keep="last")
    df = df.set_index("datetime", drop=True)
    return df


def old_add_neighbors(
    etf_executions: pd.DataFrame,
    equity_executions: pd.DataFrame,
    tolerances: list[str],
):
    """Annotate the etf execution dataframe with the indices of the neighbouring equity executions.
    Note: Building the KDTree on the equity dataframe. Blah
    """
    etf_executions = etf_executions.copy()

    etf_times = get_times(etf_executions)
    equity_times = get_times(equity_executions)
    equity_kd_tree = KDTree(equity_times, metric="l1")

    for tolerance in tolerances:
        tolerance_in_nanoseconds = str_to_nanoseconds(tolerance)
        etf_executions[f"neighbors_{tolerance}"] = equity_kd_tree.query_radius(
            etf_times, r=tolerance_in_nanoseconds
        )
        etf_executions[f"nonIso_{tolerance}"] = etf_executions[
            f"neighbors_{tolerance}"
        ].apply(lambda x: x.size > 0)

    return etf_executions


# apply this by day. maybe don't need get times as do it directly on "time" column
def add_neighbors(
    etf_executions: pd.DataFrame,
    equity_executions: pd.DataFrame,
    tolerance: str,
):
    etf_times = get_times(etf_executions)
    equity_times = get_times(equity_executions)
    equity_kd_tree = KDTree(equity_times, metric="l1")

    tolerance_in_nanoseconds = str_to_nanoseconds(tolerance)
    neighbors = equity_kd_tree.query_radius(etf_times, r=tolerance_in_nanoseconds)
    return neighbors


# TODO: datetime parser for absl
_DATE_RANGE = (dt.datetime(2021, 1, 4), dt.datetime(2021, 1, 10))
_FEATURE_FUNCTIONS = ["notional", "numTrades", "distinctTickers"]
_SAME_SIGN_OPPOSITE_SIGN = ["ss", "os"]
_BEFORE_AFTER = ["bf", "af"]

_FEATURE_NAMES = [
    "_".join(x)
    for x in it.product(_FEATURE_FUNCTIONS, _SAME_SIGN_OPPOSITE_SIGN, _BEFORE_AFTER)
]

_MARGINAL_BEFORE_AFTER = [
    "_".join(x) for x in it.product(_FEATURE_FUNCTIONS, _SAME_SIGN_OPPOSITE_SIGN)
]
_MARGINAL_SAME_SIGN_OPPOSITE_SIGN = [
    "_".join(x) for x in it.product(_FEATURE_FUNCTIONS, _BEFORE_AFTER)
]

_FEATURE_NAMES_WITH_MARGINALS = (
    _FEATURE_NAMES
    + _FEATURE_FUNCTIONS
    + _MARGINAL_BEFORE_AFTER
    + _MARGINAL_SAME_SIGN_OPPOSITE_SIGN
)
_FEATURE_NAMES_WITH_MARGINALS


_QUANTILES = [0.1, 0.9]
_LOOKBACK = 3

def evaluate_features(
    equity_executions: pd.DataFrame,
    neighbors: np.ndarray,
    etf_trade_time: float,
    etf_trade_direction: int,
):
    cols = ["time", "size", "price", "direction", "ticker", "notional"]
    features = (
        equity_executions.iloc[neighbors][cols]
        .assign(
            equityBefore=lambda _df: _df.time < etf_trade_time,
            sameSign=lambda _df: _df.direction == etf_trade_direction,
        )
        .groupby(["sameSign", "equityBefore"])
        .agg(
            notional=("notional", "sum"),
            numTrades=("size", "count"),
            distinctTickers=("ticker", "nunique"),
        )
        .stack()
        .reorder_levels([-1, 0, 1])
    )

    # TODO: turn into unittest
    # assert len(features.index.names) == 3
    # assert features.index.names[1] == "sameSign"
    # assert features.index.names[2] == "equityBefore"

    sameSignOppositeSign = features.index.levels[1].map({True: "ss", False: "os"})
    equityBeforeAfter = features.index.levels[2].map({True: "bf", False: "af"})

    features.index = features.index.set_levels(
        levels=[sameSignOppositeSign, equityBeforeAfter], level=[1, 2]
    )
    features.index = features.index.to_flat_index()
    features.index = ["_".join(x) for x in features.index]

    features.reindex(_FEATURE_NAMES, fill_value=0.0)

    return features


def add_marginals(df: pd.DataFrame) -> pd.DataFrame:
    for col_name in _FEATURE_FUNCTIONS:
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


def add_decile_tags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(deep=True)
    feature_columns = [
        col for col in df.columns if (col.startswith("_") and ("neighbors" not in col))
    ]

    for col in feature_columns:
        df[col + "_decile"] = 0
        mask = df[col] != 0
        df.loc[mask, col + "_decile"] = 1 + pd.qcut(
            df.loc[mask, col], 10, labels=False, duplicates="drop"
        )

    return df


_COLUMNS = [
    "time",
    "event",
    "order_id",
    "size",
    "price",
    "direction",
    "ask_price_1",
    "bid_price_1",
]


def main(_):
    arctic = Arctic(FLAGS.s3_uri)
    arctic_library = arctic[FLAGS.library]

    q = QueryBuilder()
    q = q[q.event.isin(EventGroup.EXECUTIONS.value)]

    equity_executions = (
        fetch_tickers(
            tickers=["FLIR", "YUM"],
            arctic_library=arctic_library,
            query_builder=q,
            columns=_COLUMNS,
            date_range=_DATE_RANGE,
        )
        .pipe(aggregate_duplicates)
        .eval("notional = price * size")
    )

    etf_executions = (
        fetch_tickers(
            tickers=[FLAGS.etf],
            arctic_library=arctic_library,
            query_builder=q,
            columns=_COLUMNS,
            date_range=_DATE_RANGE,
        )
        .pipe(aggregate_duplicates)
        .pipe(assign_mid)
    )

    # TODO: move these to unittests
    assert etf_executions.index.is_monotonic_increasing
    assert equity_executions.index.is_monotonic_increasing
    assert etf_executions.index.is_unique
    assert equity_executions.index.is_unique

    tolerance_to_features = {}
    for tolerance in FLAGS.tolerances:

        neighbors = pd.DataFrame(
            index=etf_executions.index, columns=["neighbors", "nonIso"]
        )
        neighbors["neighbors"] = add_neighbors(
            etf_executions=etf_executions,
            equity_executions=equity_executions,
            tolerance=tolerance,
        )
        neighbors["nonIso"] = neighbors["neighbors"].apply(lambda x: x.size > 0)

        features = pd.concat((etf_executions, neighbors), axis=1)[neighbors.nonIso].apply(
            lambda row: evaluate_features(
                equity_executions=equity_executions,
                neighbors=row.neighbors,
                etf_trade_direction=row.direction,
                etf_trade_time=row.time,
            ),
            axis=1,
            result_type="expand",
        )

        features = features.fillna(0.0)

        feature_to_type = {
            "numTrades": "int",
            "distinctTickers": "int",
            "notional": "float",
        }
        for feature, type_ in feature_to_type.items():
            cols = features.filter(like=feature).columns
            features[cols] = features[cols].astype(type_)

        features = add_marginals(features)

        tolerance_to_features[tolerance] = features

    output_dir = Path("./private/wip/")
    with open(output_dir / "tolerance_to_features.pkl", "wb") as f:
        pickle.dump(tolerance_to_features, f)

    write_pickle = True
    if write_pickle:
        output_dir = Path("./private/wip/")
        neighbors.to_pickle(output_dir / "neighbors.pkl")
        equity_executions.to_pickle(output_dir / "equity_executions.pkl")
        etf_executions.to_pickle(output_dir / "etf_executions.pkl")


if __name__ == "__main__":
    app.run(main)
