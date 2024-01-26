__all__ = [
    "str_to_nanoseconds",
    "col_to_dtype",
    "features",
    "all_index",
    "empty_series",
    "get_times",
    "str_to_time",
    "add_neighbors",
    "drop_all_neighbor_cols",
    "col_to_dtype_inputing_mapping",
    "multi_index_to_single_index",
    "groupby_index_to_series",
    "compute_features",
    "append_features",
    "count_non_null",
    "drop_features",
    "split_isolated_non_isolated",
    "resample_mid",
    "restrict_common_index",
    "markout_returns",
    "clip_df_times",
    "clip_for_markout",
]

import datetime as dt
import itertools as it
import typing as t
from functools import partial

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.neighbors import KDTree

from lobster_tools.preprocessing import *


def get_times(df: pd.DataFrame) -> NDArray[np.datetime64]:
    "Return numpy array of times from the index of the DataFrame."
    if df.index.values.dtype != "datetime64[ns]":
        raise TypeError("DataFrame index must be of type datetime64[ns]")
    return df.index.values.reshape(-1, 1)


def str_to_time(time: str, convert_to: str) -> int:
    return pd.Timedelta(time) / pd.Timedelta(1, unit=convert_to)


str_to_nanoseconds = lambda x: int(str_to_time(x, convert_to="ns"))


def add_neighbors(
    etf_executions: pd.DataFrame,
    equity_executions: pd.DataFrame,
    tolerances: list[str],
):
    """Annotate the etf execution dataframe with the indices of the neighbouring equity executions.
    Note: Building the KDTree on the equity dataframe. Blah
    """
    etf_executions = etf_executions.copy()

    etf_executions_times = get_times(etf_executions)
    equity_executions_times = get_times(equity_executions)
    equity_tree = KDTree(equity_executions_times, metric="l1")

    def _add_neighbors_col(etf_executions, tolerance_str):
        tolerance_in_nanoseconds = str_to_nanoseconds(tolerance_str)
        etf_executions[f"_{tolerance_str}_neighbors"] = equity_tree.query_radius(
            etf_executions_times, r=tolerance_in_nanoseconds
        )
        etf_executions[f"_{tolerance_str}_non-iso"] = etf_executions[
            f"_{tolerance_str}_neighbors"
        ].apply(lambda x: x.size > 0)

    for tolerance in tolerances:
        _add_neighbors_col(etf_executions, tolerance)

    return etf_executions


def drop_all_neighbor_cols(df: pd.DataFrame):
    "Drop neighbor columns inplace."
    neighbor_column_names = df.filter(regex="neighbors").columns
    df.drop(columns=neighbor_column_names, inplace=True)


def col_to_dtype_inputing_mapping(col, col_to_dtype_dict):
    for k, v in col_to_dtype_dict.items():
        if k in col:
            return v


col_to_dtype = partial(
    col_to_dtype_inputing_mapping,
    col_to_dtype_dict={
        "notional": pd.SparseDtype(float, 0),
        "num_trades": pd.SparseDtype(int, 0),
        "distinct_tickers": pd.SparseDtype(int, 0),
    },
)

features = ["distinct_tickers", "notional", "num_trades"]
all_index = ["_".join(t) for t in it.product(features, ["ss", "os"], ["bf", "af"])]

empty_series = pd.Series(index=all_index, dtype="Sparse[float]").fillna(0)
empty_series = pd.Series(index=all_index, dtype="float").fillna(0)


def multi_index_to_single_index(df: pd.DataFrame) -> pd.DataFrame:
    df.index = ["_".join(index_tuple) for index_tuple in df.index]
    return df


def groupby_index_to_series(df: pd.DataFrame) -> pd.Series:
    """Hierachical groupby index with one column to flattened series. Prepending the column name to the index."""
    return df.stack().reorder_levels([-1, 0, 1]).pipe(multi_index_to_single_index)


def compute_features(
    etf_trade_time,
    etf_trade_direction,
    neigh: t.Optional[np.ndarray],
    equity_executions: pd.DataFrame,
) -> pd.DataFrame:
    if neigh is None:
        return empty_series
    elif isinstance(neigh, np.ndarray):
        df = equity_executions.iloc[neigh].assign(
            bf_af=lambda df: df.index < etf_trade_time,
            ss_os=lambda df: df.direction == etf_trade_direction,
        )
        df["ss_os"] = (
            df["ss_os"].apply(lambda x: "ss" if x else "os").astype("category")
        )
        df["bf_af"] = (
            df["bf_af"].apply(lambda x: "bf" if x else "af").astype("category")
        )

        df_subset = df[["ticker", "ss_os", "bf_af", "price", "size"]]

        # notional value and num trades
        notional_and_num_trades = (
            df_subset.eval('notional = price * size.astype("int64")')
            .groupby(["ss_os", "bf_af"])
            .agg(notional=("notional", "sum"), num_trades=("size", "count"))
            .pipe(groupby_index_to_series)
        )

        # distinct tickers
        distinct_tickers = (
            df_subset.drop(columns="size")
            .groupby(["ticker", "ss_os", "bf_af"])
            .count()
            .applymap(lambda x: x > 0)
            .groupby(["ss_os", "bf_af"])
            .sum()
            .rename(columns={"price": "distinct_tickers"})
            .pipe(groupby_index_to_series)
        )

        return (
            pd.concat([notional_and_num_trades, distinct_tickers])
            .reindex(all_index)
            .fillna(0)
        )
    else:
        raise ValueError("neigh must be None or list")


def append_features(
    etf_executions: pd.DataFrame, equity_executions: pd.DataFrame
) -> pd.DataFrame:
    "Note that this function is not inplace."
    # infer tolerances from column names
    column_names = etf_executions.filter(regex="neighbors").columns.values.tolist()
    tolerances = [i.split("_")[1] for i in column_names]

    # TODO: check if its faster to partial compute features with equity executions

    features_dfs = []
    for tolerance in tolerances:
        # add_neighbors(df, equity_executions, tolerance)
        features = etf_executions.apply(
            lambda row: compute_features(
                row.name,
                row.direction,
                row[f"_{tolerance}_neighbors"],
                equity_executions=equity_executions,
            ),
            axis=1,
            result_type="expand",
        ).add_prefix(f"_{tolerance}_")

        features = features.astype({col: col_to_dtype(col) for col in features.columns})

        features_dfs += [features]

    features_df = pd.concat(features_dfs, axis=1)
    return pd.concat([etf_executions, features_df], axis=1)


def count_non_null(df, tolerance):
    return df[f"_{tolerance}_neighbors"].notnull().sum()


def drop_features(df: pd.DataFrame) -> None:
    """Drops all intermediate features, and just leaves the arbitrage tags.
    Not the nicest way. Could do better regex."""
    features_and_arb_tag = set(df.filter(regex="^_[0-9]+ms_").columns)
    arb_tag = set(df.filter(regex="arb_tag").columns)
    features = features_and_arb_tag - arb_tag
    df.drop(columns=features, inplace=True)
    return None


def split_isolated_non_isolated(
    etf_executions: pd.DataFrame, tolerance
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns a tuple of (isolated, non_isolated). For now, use deep copy, although this may not be great."""
    tolerance_str = f"_{tolerance}_neighbors"
    isolated_indices = etf_executions[tolerance_str].isna()
    return etf_executions[isolated_indices].copy(deep=True), etf_executions[
        ~isolated_indices
    ].copy(deep=True)


def resample_mid(df: pd.DataFrame, resample_freq="5T"):
    return (
        df.resample(resample_freq, label="right")
        .last()
        .eval("mid = bid_price_1 + (ask_price_1 - bid_price_1) / 2")["mid"]
    )


def restrict_common_index(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict two dataframes to their common index."""
    common_index = df1.index.intersection(df2.index)
    return df1.loc[common_index], df2.loc[common_index]


def markout_returns(
    df,  # dataframe to infer times to markout from
    markouts: list[str],  # list of markouts to compute returns for
) -> pd.DataFrame:
    return pd.DataFrame(
        index=df.index,
        data={f"_{markout}": df.index + pd.Timedelta(markout) for markout in markouts},
    )


def clip_df_times(
    df: pd.DataFrame, start: dt.time | None = None, end: dt.time | None = None
) -> pd.DataFrame:
    """Clip a dataframe or lobster object to a time range."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected a dataframe with a datetime index")

    if start and end:
        return df.iloc[(df.index.time >= start) & (df.index.time < end)]
    elif start:
        return df.iloc[df.index.time >= start]
    elif end:
        return df.iloc[df.index.time < end]
    else:
        raise ValueError("start and end cannot both be None")


def clip_for_markout(df, max_markout):
    end = (max(df.index) - pd.Timedelta(max_markout)).time()
    return clip_df_times(df, end=end)
