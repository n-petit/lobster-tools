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

flags.DEFINE_list(
    "tolerances",
    ["10ms", "20ms"],
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
    df = df.copy()
    df.reset_index(inplace=True)
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
    df.drop_duplicates(subset=["datetime"], keep="last", inplace=True)
    df.set_index("datetime", drop=True, inplace=True)
    return df


def add_neighbors(
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


# TODO: datetime parser for absl
_DATE_RANGE = (dt.datetime(2021, 1, 4), dt.datetime(2021, 1, 6))
_FACTORS = ["notional", "numTrades", "distinctTickers"]  # not a great name
_SAME_SIGN_OPPOSITE_SIGN = ["ss", "os"]
_BEFORE_AFTER = ["bf", "af"]


def get_feature_names(tolerances: list[str]):
    features = [
        "_".join(x)
        for x in it.product(
            _FACTORS, _SAME_SIGN_OPPOSITE_SIGN, _BEFORE_AFTER, tolerances
        )
    ]
    return features


# not sure about etf_trade_time type
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
    levels = [sameSignOppositeSign, equityBeforeAfter]

    features.index = features.index.set_levels(levels=levels, level=[1, 2])
    features.index = features.index.to_flat_index()
    features.index = ["_".join(x) for x in features.index]

    features.reindex(get_feature_names(tolerances=FLAGS.tolerances), fill_value=0.0)

    return features


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
        .pipe(assign_mid)
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

    annotated_etf_executions = add_neighbors(
        etf_executions=etf_executions,
        equity_executions=equity_executions,
        tolerances=FLAGS.tolerances,
    )
    logging.info(annotated_etf_executions.head())

    features = get_feature_names(tolerances=FLAGS.tolerances)
    logging.info(features)

    annotated_etf_executions = annotated_etf_executions.assign(
        **{feature: 0.0 for feature in features}
    )
    logging.info(annotated_etf_executions.dtypes)

    write_pickle = True
    if write_pickle:
        output_dir = Path("./private/wip/")
        equity_executions.to_pickle(output_dir / "equity_executions.pkl")
        etf_executions.to_pickle(output_dir / "etf_executions.pkl")
        annotated_etf_executions.to_pickle(output_dir / "annotated_etf_executions.pkl")


if __name__ == "__main__":
    app.run(main)
