# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/00_preprocessing.ipynb.

# %% auto 0
__all__ = ['LoadType', 'TickerTypes', 'Event', 'EventGroup', 'Direction', 'Data', 'aggregate_duplicates_', 'Lobster',
           'load_lobster']

# %% ../notebooks/00_preprocessing.ipynb 4
import enum
import glob
from typing import Optional, Literal, get_args
from dataclasses import dataclass
import re
import pandas as pd
import numpy as np
import os
import datetime
from pprint import pprint
from functools import partial

# %% ../notebooks/00_preprocessing.ipynb 5
@enum.unique
class Event(enum.Enum):
    "A class to represent the event type of a LOBSTER message."
    UNKNOWN = 0
    SUBMISSION = 1
    CANCELLATION = 2
    DELETION = 3
    EXECUTION = 4
    HIDDEN_EXECUTION = 5
    CROSS_TRADE = 6
    OTHER = 8
    TRADING_HALT = 9
    RESUME_QUOTE = 10
    TRADING_RESUME = 11


@enum.unique
class EventGroup(enum.Enum):
    EXECUTIONS = [Event.EXECUTION.value, Event.HIDDEN_EXECUTION.value]
    HALTS = [
        Event.TRADING_HALT.value,
        Event.RESUME_QUOTE.value,
        Event.TRADING_RESUME.value,
    ]
    CANCELLATIONS = [Event.CANCELLATION.value, Event.DELETION.value]

# %% ../notebooks/00_preprocessing.ipynb 11
@enum.unique
class Direction(enum.Enum):
    BUY = 1
    SELL = -1

# %% ../notebooks/00_preprocessing.ipynb 15
LoadType = Literal["both", "messages", "book"]
TickerTypes = Literal[None, "equity", "etf"]

# %% ../notebooks/00_preprocessing.ipynb 16
# | code-fold: true
@dataclass
class Data:
    directory_path: str = "/home/petit/Documents/code/lobster-tools/data"  # path to data
    ticker: str = "AMZN"
    date_range: Optional[str | tuple[str, str]] = None
    levels: Optional[int] = None
    nrows: Optional[int] = None
    load: LoadType = "messages"
    add_ticker: bool = True
    ticker_type: TickerTypes = None
    clip_trading_hours: bool = True
    aggregate_duplicates: bool = True

    def __post_init__(self) -> None:
        assert self.load in get_args(LoadType)
        assert self.ticker_type in get_args(TickerTypes)

        # ticker path
        tickers = glob.glob(f"{self.directory_path}/*")
        ticker_path = [t for t in tickers if self.ticker in t]
        assert len(ticker_path) == 1
        self.ticker_path = ticker_path[0]

        # levels
        if not self.levels:
            self.levels = int(self.ticker_path.split("_")[-1])
            assert self.levels >= 1

        # infer date range from ticker folder name
        if not self.date_range:
            self.date_range = tuple(re.findall(pattern=r"\d\d\d\d-\d\d-\d\d", string=self.ticker_path))
            assert len(self.date_range) == 2

        # book and message paths
        tickers = glob.glob(f"{self.ticker_path}/*")
        tickers_end = list(map(os.path.basename, tickers))

        if isinstance(self.date_range, tuple):
            # get all dates in folder
            dates = set([re.findall(pattern=r"\d\d\d\d-\d\d-\d\d", string=file)[0] for file in tickers_end])
            # filter for dates within specified range
            dates = sorted(
                list(
                    filter(
                        lambda date: self.date_range[0] <= date <= self.date_range[1],
                        dates,
                    )
                )
            )

            self.dates = dates
            self.date_range = (min(self.dates), max(self.dates))

        elif isinstance(self.date_range, str):
            self.dates, self.date_range = [self.date_range], (
                self.date_range,
                self.date_range,
            )

        # messages and book filepath dictionaries
        def _create_date_to_path_dict(keyword: str) -> dict:
            filter_keyword_tickers = list(filter(lambda x: keyword in x, tickers_end))
            date_path_dict = {}
            for date in self.dates:
                filter_date_tickers = list(filter(lambda x: date in x, filter_keyword_tickers))
                assert len(filter_date_tickers) == 1
                date_path_dict[date] = os.path.join(self.ticker_path, filter_date_tickers[0])
            return date_path_dict

        self.book_paths = _create_date_to_path_dict("book")
        self.messages_paths = _create_date_to_path_dict("message")

# %% ../notebooks/00_preprocessing.ipynb 17
def aggregate_duplicates_(df: pd.DataFrame) -> pd.DataFrame:
    subset = list(df.columns)
    subset.remove("size")
    subset.insert(0, "datetime")

    duplicates_index = df.reset_index().duplicated(subset=subset, keep=False).values
    duplicates = df[duplicates_index]
    no_duplicates = df[~duplicates_index]

    assert no_duplicates.shape[0] + duplicates.shape[0] == df.shape[0]

    aggregated_duplicates = duplicates.groupby(by=[duplicates.index, "direction"]).agg({"size": "sum"})
    aggregated_duplicates = pd.merge(duplicates, aggregated_duplicates, on="datetime", suffixes=("_original", "_agg"))
    aggregated_duplicates.drop(columns=["size_original"], inplace=True)
    aggregated_duplicates.rename(columns={"size_agg": "size"}, inplace=True)
    aggregated_duplicates.drop_duplicates(keep="first", inplace=True)

    df = pd.concat([no_duplicates, aggregated_duplicates])
    df.sort_index()
    return df

# %% ../notebooks/00_preprocessing.ipynb 19
# | code-fold: true
@dataclass
class Lobster:
    "Lobster data class for a single symbol of Lobster data."
    data: Data | None = None

    def __post_init__(self):
        if self.data is None:
            self.data = Data()
        
        if self.data.load in ["messages", "both"]:
            dfs = []
            for date, filepath in self.data.messages_paths.items():
                # load messages
                df = pd.read_csv(
                    filepath,
                    header=None,
                    nrows=self.data.nrows,
                    usecols=list(range(6)),
                    names=["time", "event", "order_id", "size", "price", "direction"],
                    index_col=False,
                    dtype={
                        "time": "float64",
                        "event": "uint8",
                        "price": "int64",
                        "direction": "int8",
                        "order_id": "uint32",
                        "size": "uint64",
                    },
                )

                # set index as datetime
                df["datetime"] = pd.to_datetime(date, format="%Y-%m-%d") + df.time.apply(lambda x: pd.to_timedelta(x, unit="s"))
                df.set_index("datetime", drop=True, inplace=True)
                df.drop(columns="time", inplace=True)
                dfs.append(df)
            df = pd.concat(dfs)

            # use 0 as NaN for price, size and direction
            assert df.loc[df.event.eq(Event.TRADING_HALT.value), "direction"].eq(-1).all()
            df.loc[df.event.eq(Event.TRADING_HALT.value), "direction"] = 0

            # process trading halts
            def _trading_halt_type(price):
                return {
                    -1: Event.TRADING_HALT.value,
                    0: Event.RESUME_QUOTE.value,
                    1: Event.TRADING_RESUME.value,
                }[price]

            df.loc[df.event.eq(Event.TRADING_HALT.value), "event"] = df.loc[df.event.eq(Event.TRADING_HALT.value), "price"].apply(
                lambda x: _trading_halt_type(x)
            )

            df.loc[
                df.event.isin(EventGroup.HALTS.value),
                ["order_id", "size", "price"],
            ] = [
                0,
                0,
                np.nan,
            ]

            # set price in dollars
            df.price = df.price.apply(lambda x: x / 10_000).astype("float64")

            if self.data.add_ticker:
                df = df.assign(ticker=self.data.ticker).astype({"ticker": "category"})

            if self.data.ticker_type:
                assert self.data.ticker_type in [
                    "equity",
                    "etf",
                ], "ticker_type must be either `equity` or `etf`"
                df = df.assign(ticker_type=self.data.ticker_type).astype(
                    dtype={"ticker_type": pd.CategoricalDtype(categories=["equity", "etf"])}
                )

            self.messages = df

        if self.data.load in ["book", "both"]:
            col_names = []
            for level in range(1, self.data.levels + 1):
                for col_type in ["ask_price", "ask_size", "bid_price", "bid_size"]:
                    col_name = f"{col_type}_{level}"
                    col_names.append(col_name)

            # for now just use float64
            # col_dtypes = {
            #     col_name: pd.Int64Dtype() if ("size" in col_name) else "float"
            #     for col_name in col_names
            # }

            dfs = []
            for filename in self.data.book_paths.values():
                df = pd.read_csv(
                    filename,
                    header=None,
                    nrows=self.data.nrows,
                    usecols=list(range(4 * self.data.levels)),
                    names=col_names,
                    dtype="float64",
                    na_values=[-9999999999, 9999999999, 0],
                )

                dfs.append(df)
            df = pd.concat(dfs)
            
            df.set_index(self.messages.index, inplace=True, drop=True)

            price_cols = df.columns.str.contains("price")
            df.loc[:, price_cols] = df.loc[:, price_cols].apply(lambda x: x / 10_000)

            self.book = df

        # data cleaning on messages done only now, as book infers times from messages file
        # aggregate duplicates 
        if self.data.aggregate_duplicates:
            self.messages = aggregate_duplicates_(self.messages)

        # clip messages to trading hours (from 9:30 to 4:30)
        if self.data.clip_trading_hours:
            df = df.iloc[(df.index.time >= datetime.time(9, 30)) & (df.index.time < datetime.time(16, 30))]

            
    def __repr__(self) -> str:
        return f"Lobster data for ticker: {self.data.ticker} for date range: {self.data.date_range[0]} to {self.data.date_range[1]}."

# %% ../notebooks/00_preprocessing.ipynb 21
def load_lobster(**kwargs):
    """Load `Lobster` object from csv data."""
    data = Data(**kwargs)
    lobster = Lobster(data)

    return lobster
