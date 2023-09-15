# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/01_preprocessing.ipynb.

# %% auto 0
__all__ = ['Event', 'EventGroup', 'Direction', 'Data', 'clip_df_times', 'Lobster', 'load_lobster', 'load_lobsters']

# %% ../notebooks/01_preprocessing.ipynb 5
import datetime
import enum
import glob
import os
import re
from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional, Self, Protocol, runtime_checkable, get_args

import numpy as np
import pandas as pd

# %% ../notebooks/01_preprocessing.ipynb 6
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
    TRADING_HALT = 7
    RESUME_QUOTE = 10
    TRADING_RESUME = 11


@enum.unique
class EventGroup(enum.Enum):
    EXECUTIONS = [Event.EXECUTION.value, Event.HIDDEN_EXECUTION.value, Event.CROSS_TRADE.value]
    HALTS = [
        Event.TRADING_HALT.value,
        Event.RESUME_QUOTE.value,
        Event.TRADING_RESUME.value,
    ]
    CANCELLATIONS = [Event.CANCELLATION.value, Event.DELETION.value]

# %% ../notebooks/01_preprocessing.ipynb 12
@enum.unique
class Direction(enum.Enum):
    BUY = 1
    SELL = -1

# %% ../notebooks/01_preprocessing.ipynb 16
# | code-fold: true
@dataclass
class Data:
    directory_path: str | None = None  # path to data
    ticker: str | None = None  # ticker name
    date_range: Optional[str | tuple[str, str]] = None
    levels: Optional[int] = None
    nrows: Optional[int] = None
    load: Literal["both", "messages", "book"] = "both"
    add_ticker_column: bool = False
    ticker_type: Literal[None, "equity", "etf"] = None
    clip_trading_hours: bool = True
    aggregate_duplicates: bool = True

    def __post_init__(self) -> None:
        if self.directory_path is None:
            self.directory_path = os.getenv("LOBSTER_DATA_PATH", "../data")
        if self.ticker is None:
            self.ticker = os.getenv("DEFAULT_TICKER", "AMZN")

        # TODO: do this better, maybe pydantic, maybe a decorator, or maybe with
        # LoadType = Literal["both", "messages", "book"]
        # TickerTypes = Literal[None, "equity", "etf"]
        # assert self.ticker_type in get_args(TickerTypes)
        if self.load not in ("both", "messages", "book"):
            raise ValueError(f"Invalid load type: {self.load}")
        if self.ticker_type not in (None, "equity", "etf"):
            raise ValueError(f"Invalid ticker type: {self.ticker_type}")

        # ticker path
        tickers = glob.glob(f"{self.directory_path}/*")
        # print(self.ticker)
        ticker_path = [t for t in tickers if self.ticker in t]
        if len(ticker_path) != 1:
            raise ValueError(f"Expected exactly 1 directory with name {self.ticker}")
        # assert len(ticker_path) == 1
        self.ticker_path = ticker_path[0]

        # levels
        if not self.levels:
            self.levels = int(self.ticker_path.split("_")[-1])

            if self.levels < 1:
                raise ValueError(f"Invalid number of levels: {self.levels}")
            # assert self.levels >= 1

        # infer date range from ticker folder name
        if not self.date_range:
            self.date_range = tuple(re.findall(pattern=r"\d\d\d\d-\d\d-\d\d", string=self.ticker_path))
            if len(self.date_range) != 2:
                raise ValueError(f"Expected exactly 2 dates in regex match of in {self.ticker_path}")
            # assert len(self.date_range) == 2

        # book and message paths
        tickers = glob.glob(f"{self.ticker_path}/*")
        tickers_end = list(map(os.path.basename, tickers))

        if isinstance(self.date_range, tuple):
            # get all dates in folder
            dates = {re.findall(pattern=r"\d\d\d\d-\d\d-\d\d", string=file)[0] for file in tickers_end}
            # dates = set([re.findall(pattern=r"\d\d\d\d-\d\d-\d\d", string=file)[0] for file in tickers_end])
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
                if len(filter_date_tickers) != 1:
                    raise ValueError(f"Expected exactly 1 match for {date}")
                # assert len(filter_date_tickers) == 1
                date_path_dict[date] = os.path.join(self.ticker_path, filter_date_tickers[0])
            return date_path_dict

        self.book_paths = _create_date_to_path_dict("book")
        self.messages_paths = _create_date_to_path_dict("message")

        if self.load in ("book", "both"):
            self.load_book = True
        if self.load in ("messages", "both"):
            self.load_messages = True

# %% ../notebooks/01_preprocessing.ipynb 17
def _aggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.reset_index(inplace=True)
    duplicates = df.duplicated(subset=df.columns.difference(["size"]), keep=False)
    df.loc[duplicates, "size"] = df.loc[duplicates, "size"].groupby(df.loc[duplicates].datetime).transform("sum")
    df.drop_duplicates(subset=df.columns.difference(["size"]), inplace=True)
    df.set_index("datetime", inplace=True, drop=True)
    return df

# %% ../notebooks/01_preprocessing.ipynb 19
def clip_df_times(df: pd.DataFrame, start: datetime.time | None = None, end: datetime.time | None = None) -> pd.DataFrame:
    """Clip a dataframe or lobster object to a time range."""
    # TODO: improve this function? with the 4 if statements lol, i guess i could clip the index twice and have two if statements
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


_clip_df_trading_hours = partial(clip_df_times, start=datetime.time(9, 30), end=datetime.time(16, 0))

# %% ../notebooks/01_preprocessing.ipynb 21
def clip_df_times(df: pd.DataFrame, start: datetime.time | None = None, end: datetime.time | None = None) -> pd.DataFrame:
    """Clip a dataframe or lobster object to a time range."""
    # TODO: improve this function? with the 4 if statements lol, i guess i could clip the index twice and have two if statements
    if start and end:
        return df.iloc[(df.index.time >= start) & (df.index.time < end)]
    elif start:
        return df.iloc[df.index.time >= start]
    elif end:
        return df.iloc[df.index.time < end]
    else:
        raise ValueError("start and end cannot both be None")


_clip_df_trading_hours = partial(clip_df_times, start=datetime.time(9, 30), end=datetime.time(16, 0))

# %% ../notebooks/01_preprocessing.ipynb 22
# | code-fold: true
@dataclass
class Lobster:
    "Lobster data class for a single symbol of Lobster data."
    data: Data | None = None

    def __post_init__(self):
        if self.data is None:
            self.data = Data()

        if self.data.load_messages:
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

            if not df.loc[df.event.eq(Event.TRADING_HALT.value), "direction"].eq(-1).all():
                raise ValueError("All trading halts must have direction -1")
            # assert df.loc[df.event.eq(Event.TRADING_HALT.value), "direction"].eq(-1).all()
            df.loc[df.event.eq(Event.TRADING_HALT.value), "direction"] = 0

            # process trading halts
            def _trading_halt_type(price):
                return {
                    -1: Event.TRADING_HALT.value,
                    0: Event.RESUME_QUOTE.value,
                    1: Event.TRADING_RESUME.value,
                }[price]

            # TODO: change from apply?
            df.loc[df.event.eq(Event.TRADING_HALT.value), "event"] = df.loc[df.event.eq(Event.TRADING_HALT.value), "price"].apply(_trading_halt_type)

            # use 0 as NaN for size and direction
            df.loc[
                df.event.isin(EventGroup.HALTS.value),
                ["order_id", "size", "price"],
            ] = [0, 0, np.nan]

            # set price in dollars
            df.price = df.price.apply(lambda x: x / 10_000).astype("float64")

            if self.data.ticker_type:
                # TODO change to get Literal Values from args?

                if self.data.ticker_type not in [
                    "equity",
                    "etf",
                ]:
                    raise ValueError("ticker_type must be either `equity` or `etf`")
                # assert self.data.ticker_type in [
                #     "equity",
                #     "etf",
                # ], "ticker_type must be either `equity` or `etf`"
                df = df.assign(ticker_type=self.data.ticker_type).astype(
                    dtype={"ticker_type": pd.CategoricalDtype(categories=["equity", "etf"])}
                )

            self.messages = df

        if self.data.load_book:
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
        # TODO: think if leaving bool flags good idea
        if self.data.aggregate_duplicates:
            self.aggregate_duplicates()
        if self.data.clip_trading_hours:
            self.clip_trading_hours()
        if self.data.add_ticker_column:
            self.add_ticker_column()

    def clip_trading_hours(self) -> Self:
        if hasattr(self, "book"):
            self.book = _clip_df_trading_hours(self.book)
        if hasattr(self, "messages"):
            self.messages = _clip_df_trading_hours(self.messages)
        return self

    def aggregate_duplicates(self) -> Self:
        self.messages = _aggregate_duplicates(self.messages)
        return self

    # TODO: write decorator to simplify the "both", "messages", "book" logic that is common to a few methods
    def add_ticker_column(self, to: Literal["both", "messages", "book"] = "messages") -> Self:
        if to in ("both", "messages"):
            self.messages = self.messages.assign(ticker=self.data.ticker).astype({"ticker": "category"})
        if to in ("both", "book"):
            self.book = self.book.assign(ticker=self.data.ticker).astype({"ticker": "category"})
        return self

    def __repr__(self) -> str:
        return f"Lobster data for ticker: {self.data.ticker} for date range: {self.data.date_range[0]} to {self.data.date_range[1]}."

# %% ../notebooks/01_preprocessing.ipynb 28
def load_lobster(**kwargs):
    """Load `Lobster` object from csv data."""
    # TODO remove this function and turn Lobster into callable class
    data = Data(**kwargs)
    lobster = Lobster(data)

    return lobster

# %% ../notebooks/01_preprocessing.ipynb 29
def load_lobsters(**kwargs):
    """Load multiple `Lobster` objects into list."""
    if not isinstance(kwargs["ticker"], list):
        raise ValueError("load lobsters is used for loading multiple tickers")
    # assert isinstance(kwargs["ticker"], list), "load lobsters is used for loading multiple tickers"
    tickers = kwargs.pop("ticker")

    lobsters = []
    for ticker in tickers:
        data = Data(ticker=ticker, **kwargs)
        lobsters += [Lobster(data)]

    return lobsters
