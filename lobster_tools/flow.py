import datetime as dt
import itertools as it
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging
from arcticdb import Arctic, QueryBuilder
from sklearn.neighbors import KDTree

from lobster_tools.config import ETF_TO_EQUITIES
from lobster_tools.preprocessing import Event, EventGroup

import lobster_tools.config  # noqa: F401

flags.DEFINE_list(
    "date_range",
    None,
    "Date range.",
)

FLAGS = flags.FLAGS

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


def aggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()

    duplicates = df[df.duplicated(subset=["datetime", "direction"], keep=False)]
    duplicates = duplicates.eval("notional = price * size")

    grouped_df = duplicates.groupby(["datetime", "direction"])

    # TODO: only average price if there are multiple prices
    total_size = grouped_df["size"].transform("sum")
    total_notional = grouped_df["notional"].transform("sum")
    average_price = total_notional / total_size

    df.loc[duplicates.index, "size"] = total_size
    df.loc[duplicates.index, "price"] = average_price

    _MERGED_ORDER_ID = -2  # -1 already used for auction
    df.loc[duplicates.index, ["event", "order_id"]] = [
        Event.AGGREGATED.value,
        _MERGED_ORDER_ID,
    ]
    df = df.drop_duplicates(subset=["datetime"], keep="last")
    df = df.set_index("datetime", drop=True)
    assert df.index.is_unique

    return df


_FEATURE_FUNCTIONS = ["notional", "numTrades", "distinctTickers"]
_SAME_SIGN_OPPOSITE_SIGN = ["ss", "os"]
_BEFORE_AFTER = ["bf", "af"]


def _combine_features(*args):
    return ["_".join(x) for x in it.product(*args)]


_FEATURE_NAMES = _combine_features(
    _FEATURE_FUNCTIONS, _SAME_SIGN_OPPOSITE_SIGN, _BEFORE_AFTER
)
_MARGINAL_BEFORE_AFTER = _combine_features(_FEATURE_FUNCTIONS, _SAME_SIGN_OPPOSITE_SIGN)
_MARGINAL_SAME_SIGN_OPPOSITE_SIGN = _combine_features(_FEATURE_FUNCTIONS, _BEFORE_AFTER)

_FEATURE_NAMES_WITH_MARGINALS = (
    _FEATURE_NAMES
    + _FEATURE_FUNCTIONS
    + _MARGINAL_BEFORE_AFTER
    + _MARGINAL_SAME_SIGN_OPPOSITE_SIGN
)


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
    for col in _FEATURE_FUNCTIONS:
        # marginalise over bf/af and ss/os
        df[col] = (
            df[col + "_ss_af"]
            + df[col + "_ss_bf"]
            + df[col + "_os_af"]
            + df[col + "_os_bf"]
        )

        # marginalise over bf/af
        df[col + "_ss"] = df[col + "_ss_af"] + df[col + "_ss_bf"]
        df[col + "_os"] = df[col + "_os_af"] + df[col + "_os_bf"]

        # marginalise over ss/os
        df[col + "_af"] = df[col + "_ss_af"] + df[col + "_os_af"]
        df[col + "_bf"] = df[col + "_ss_bf"] + df[col + "_os_bf"]
    return df


def main(_):
    # set output directory and logging
    output_dir = Path(FLAGS.output_dir)
    etf_dir = output_dir / "etf" / FLAGS.etf
    etf_dir.mkdir(parents=True)

    logging.get_absl_handler().use_absl_log_file(log_dir=etf_dir)
    sys.stderr = logging.get_absl_handler().python_handler.stream

    # parse flags
    logging.info(f"FLAGS={FLAGS}")
    epsilons = FLAGS.epsilons
    date_range = (
        tuple(
            dt.datetime.strptime(date, "%Y-%m-%d").date() for date in FLAGS.date_range
        )
        if FLAGS.date_range
        else None
    )
    equities = ETF_TO_EQUITIES[FLAGS.etf]

    arctic = Arctic(FLAGS.s3_uri)
    arctic_library = arctic[FLAGS.library]

    logging.info(f"Equities listed in ETF_TO_EQUITIES={equities}")
    # restrict to equities present in the library
    equities = [x for x in equities if x in arctic_library.list_symbols()]

    # fetch data
    logging.info(f"Fetching data for ticker={equities}")
    q = QueryBuilder()
    q = q[q.event.isin(EventGroup.EXECUTIONS.value)]

    equity_executions = (
        pd.concat(
            (
                arctic_library.read(
                    symbol=ticker,
                    columns=_COLUMNS,
                    query_builder=q,
                    date_range=date_range,
                )
                .data.pipe(aggregate_duplicates)
                .assign(
                    ticker=ticker,
                )
                .eval("mid = (bid_price_1 + ask_price_1) / 2")
                .eval("notional = price * size")
                for ticker in equities
            )
        )
        .sort_index()
        .astype({"ticker": "category"})
    )

    logging.info(f"Fetching data for ticker={FLAGS.etf}")
    etf_executions = (
        arctic_library.read(
            symbol=FLAGS.etf,
            query_builder=q,
            columns=_COLUMNS,
            date_range=date_range,
        )
        .data.pipe(aggregate_duplicates)
        .eval("mid = (bid_price_1 + ask_price_1) / 2")
        .eval("notional = price * size")
    )

    assert etf_executions.index.is_monotonic_increasing
    assert equity_executions.index.is_monotonic_increasing
    assert etf_executions.index.is_unique
    assert equity_executions.index.is_unique

    # save trade data to disk
    equity_executions.to_pickle(etf_dir / "equity_executions.pkl")
    etf_executions.to_pickle(etf_dir / "etf_executions.pkl")

    # compute kd tree
    etf_times = etf_executions.index.values.reshape(-1, 1)
    equity_times = equity_executions.index.values.reshape(-1, 1)

    kd_tree = KDTree(equity_times, metric="l1")

    # compute features
    for epsilon in epsilons:
        logging.info(f"Computing features for epsilon={str(epsilon)}")

        neighbors = pd.DataFrame(
            index=etf_executions.index, columns=["neighbors", "nonIso"]
        )

        r = int(pd.Timedelta(epsilon) / pd.Timedelta(1, unit="ns"))
        neighbors["neighbors"] = kd_tree.query_radius(etf_times, r=r)
        neighbors["nonIso"] = neighbors["neighbors"].apply(lambda x: x.size > 0)

        features = pd.concat((etf_executions, neighbors), axis=1)[
            neighbors.nonIso
        ].apply(
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
        for feature, dtype in feature_to_type.items():
            cols = features.filter(like=feature).columns
            features[cols] = features[cols].astype(dtype)

        features = add_marginals(features)

        # save to disk
        eps_dir = etf_dir / "epsilon" / epsilon
        eps_dir.mkdir(parents=True)

        neighbors.to_pickle(eps_dir / "neighbors.pkl")
        features.to_pickle(eps_dir / "features.pkl")


if __name__ == "__main__":
    app.run(main)
