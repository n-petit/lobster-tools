# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/01_preprocessing.ipynb.

# %% auto 0
__all__ = ['Event', 'EventGroup', 'Direction', 'Data', 'clip_times', 'Lobster', 'MPLobster', 'FolderInfo', 'infer_ticker_dict']

# %% ../notebooks/01_preprocessing.ipynb 5
import datetime
import enum
import gc
import glob
import multiprocessing as mp
import os
import re
import typing as t
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

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
    ORIGINAL_TRADING_HALT = 7
    OTHER = 8
    TRADING_HALT = 9
    RESUME_QUOTE = 10
    TRADING_RESUME = 11


@enum.unique
class EventGroup(enum.Enum):
    """Note that the `EXECUTIONS` group does not contain `Event.CROSS_TRADE`."""

    EXECUTIONS = [
        Event.EXECUTION.value,
        Event.HIDDEN_EXECUTION.value,
        # Event.CROSS_TRADE.value,
    ]
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
    date_range: t.Optional[str | tuple[str, str]] = None
    levels: t.Optional[int] = None
    nrows: t.Optional[int] = None
    load: t.Literal["both", "messages", "book"] = "both"
    add_ticker_column: bool = False
    ticker_type: t.Literal[None, "equity", "etf"] = None
    clip_trading_hours: bool = True
    aggregate_duplicates: bool = True

    def __post_init__(self) -> None:
        if self.directory_path is None:
            self.directory_path = os.getenv("LOBSTER_DATA_PATH", "../data")
        if self.ticker is None:
            self.ticker = os.getenv("DEFAULT_TICKER", "AMZN")

        # TODO: do this better, maybe pydantic, maybe a decorator, or maybe with
        # LoadType = t.Literal["both", "messages", "book"]
        # TickerTypes = t.Literal[None, "equity", "etf"]
        # assert self.ticker_type in get_args(TickerTypes)
        if self.load not in ("both", "messages", "book"):
            raise ValueError(f"Invalid load type: {self.load}")
        if self.ticker_type not in (None, "equity", "etf"):
            raise ValueError(f"Invalid ticker type: {self.ticker_type}")

        # ticker path
        tickers = glob.glob(f"{self.directory_path}/*")
        # ticker_path = [t for t in tickers if self.ticker in t]
        ticker_path = [
            t for t in tickers if os.path.basename(t).startswith(f"{self.ticker}_")
        ]

        if len(ticker_path) != 1:
            raise ValueError(f"Expected exactly 1 directory with name {self.ticker}")
        self.ticker_path = ticker_path[0]

        # levels
        if self.levels is None:
            self.levels = int(self.ticker_path.split("_")[-1])

            if self.levels < 1:
                raise ValueError(f"Invalid number of levels: {self.levels}")
        if self.levels is None:
            raise ValueError("Unable to infer levels from folder structure.")
        self.levels = t.cast(int, self.levels)

        # infer date range from ticker folder name
        if not self.date_range:
            self.date_range = tuple(
                re.findall(pattern=r"\d\d\d\d-\d\d-\d\d", string=self.ticker_path)
            )
            if len(self.date_range) != 2:
                raise ValueError(
                    f"Expected exactly 2 dates in regex match of in {self.ticker_path}"
                )

        # book and message paths
        tickers = glob.glob(f"{self.ticker_path}/*")
        tickers_end = list(map(os.path.basename, tickers))

        if isinstance(self.date_range, tuple):
            # get all dates in folder
            dates = {
                re.findall(pattern=r"\d\d\d\d-\d\d-\d\d", string=file)[0]
                for file in tickers_end
            }
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
            self.dates, self.date_range = (
                [self.date_range],
                (
                    self.date_range,
                    self.date_range,
                ),
            )

        # messages and book filepath dictionaries
        def _create_date_to_path_dict(keyword: str) -> dict:
            filter_keyword_tickers = list(filter(lambda x: keyword in x, tickers_end))
            date_path_dict = {}
            for date in self.dates:
                filter_date_tickers = list(
                    filter(lambda x: date in x, filter_keyword_tickers)
                )
                if len(filter_date_tickers) != 1:
                    raise ValueError(f"Expected exactly 1 match for {date}")
                date_path_dict[date] = os.path.join(
                    self.ticker_path, filter_date_tickers[0]
                )
            return date_path_dict

        self.book_paths = _create_date_to_path_dict("book")
        self.messages_paths = _create_date_to_path_dict("message")

        self.load_book = False
        self.load_messages = False
        if self.load in ("book", "both"):
            self.load_book = True
        if self.load in ("messages", "both"):
            self.load_messages = True

# %% ../notebooks/01_preprocessing.ipynb 17
def _aggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.reset_index(inplace=True)
    duplicates = df.duplicated(subset=df.columns.difference(["size"]), keep=False)
    df.loc[duplicates, "size"] = (
        df.loc[duplicates, "size"].groupby(df.loc[duplicates].datetime).transform("sum")
    )
    df.drop_duplicates(subset=df.columns.difference(["size"]), inplace=True)
    df.set_index("datetime", inplace=True, drop=True)
    return df

# %% ../notebooks/01_preprocessing.ipynb 19
def clip_times(
    df: pd.DataFrame,
    start: datetime.time | None = None,
    end: datetime.time | None = None,
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


_clip_to_trading_hours = partial(
    clip_times, start=datetime.time(9, 30), end=datetime.time(16, 0)
)

# %% ../notebooks/01_preprocessing.ipynb 20
# | code-fold: true
@dataclass
class Lobster:
    "Lobster data class for a single symbol of Lobster data."

    data: Data | None = None

    def process_messages(self, date, filepath):
        df = pd.read_csv(
            filepath,
            header=None,
            nrows=self.data.nrows,
            usecols=list(range(6)),
            names=["time", "event", "order_id", "size", "price", "direction"],
            index_col=False,
            dtype={
                "time": "float64",
                "event": "int8",
                "price": "int64",
                "direction": "int8",
                "order_id": "int32",
                "size": "int64",
            },
        )

        # set index as datetime
        # df.rename(columns={'time':'seconds_since_midnight'})
        df["datetime"] = pd.to_datetime(date, format="%Y-%m-%d") + df.time.apply(
            lambda x: pd.to_timedelta(x, unit="s")
        )
        df.set_index("datetime", drop=True, inplace=True)
        return df

    def __post_init__(self):
        if self.data is None:
            self.data = Data()

        if self.data.load_messages:
            # dfs = []
            # for date, filepath in self.data.messages_paths.items():
            #     # load messages
            #     df = pd.read_csv(
            #         filepath,
            #         header=None,
            #         nrows=self.data.nrows,
            #         usecols=list(range(6)),
            #         names=["time", "event", "order_id", "size", "price", "direction"],
            #         index_col=False,
            #         dtype={
            #             "time": "float64",
            #             "event": "int8",
            #             "price": "int64",
            #             "direction": "int8",
            #             "order_id": "int32",
            #             "size": "int64",
            #         },
            #     )

            #     # set index as datetime
            #     # df.rename(columns={'time':'seconds_since_midnight'})
            #     df["datetime"] = pd.to_datetime(date, format="%Y-%m-%d") + df.time.apply(lambda x: pd.to_timedelta(x, unit="s"))
            #     df.set_index("datetime", drop=True, inplace=True)
            #     dfs.append(df)
            # df = pd.concat(dfs)

            # refactor for memory use
            dfs = (
                self.process_messages(date=date, filepath=filepath)
                for date, filepath in self.data.messages_paths.items()
            )
            df = pd.concat(dfs)

            # direction for cross trades is set to zero, and order_id is left unchanged
            if (
                not df.loc[df.event.eq(Event.CROSS_TRADE.value), "direction"]
                .eq(-1)
                .all()
            ):
                raise ValueError("All cross trades must have direction -1")
            df.loc[df.event.eq(Event.CROSS_TRADE.value), "direction"] = 0

            # seems as though this is not true?
            # if not df.loc[df.event.eq(Event.CROSS_TRADE.value), "order_id"].eq(-1).all():
            #     raise ValueError("All cross trades must have order_id -1")

            if (
                not df.loc[df.event.eq(Event.ORIGINAL_TRADING_HALT.value), "direction"]
                .eq(-1)
                .all()
            ):
                raise ValueError("All trading halts must have direction -1")
            df.loc[df.event.eq(Event.ORIGINAL_TRADING_HALT.value), "direction"] = 0

            # # process trading halts and map to new trading halts
            def _trading_halt_type(price):
                return {
                    -1: Event.TRADING_HALT.value,
                    0: Event.RESUME_QUOTE.value,
                    1: Event.TRADING_RESUME.value,
                }[price]

            df.loc[df.event.eq(Event.ORIGINAL_TRADING_HALT.value), "event"] = df.loc[
                df.event.eq(Event.ORIGINAL_TRADING_HALT.value), "price"
            ].apply(_trading_halt_type)

            # implentation of above without apply
            # trading_halt_mask = (df.event == Event.TRADING_HALT.value)
            # halt_type_mapping = {
            #     -1: Event.TRADING_HALT.value,
            #     0: Event.RESUME_QUOTE.value,
            #     1: Event.TRADING_RESUME.value,
            # }
            # df.loc[trading_halt_mask, "event"] = df.loc[trading_halt_mask, "price"].map(halt_type_mapping)

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
                    dtype={
                        "ticker_type": pd.CategoricalDtype(categories=["equity", "etf"])
                    }
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
                    na_values=["-9999999999", "9999999999", "0"],
                )

                dfs.append(df)
            df = pd.concat(dfs)

            df.set_index(self.messages.index, inplace=True, drop=True)

            price_cols = df.columns.str.contains("price")
            df.loc[:, price_cols] = df.loc[:, price_cols] / 10_000

            self.book = df

        # data cleaning on messages done only now, as book infers times from messages file
        # TODO: think if leaving bool flags good idea
        if self.data.aggregate_duplicates:
            self.aggregate_duplicates()
        if self.data.clip_trading_hours:
            self.clip_trading_hours()
        if self.data.add_ticker_column:
            self.add_ticker_column()

    def clip_trading_hours(self) -> t.Self:
        if hasattr(self, "book"):
            self.book = _clip_to_trading_hours(self.book)
        if hasattr(self, "messages"):
            self.messages = _clip_to_trading_hours(self.messages)
        return self

    def aggregate_duplicates(self) -> t.Self:
        self.messages = _aggregate_duplicates(self.messages)
        return self

    # TODO: write decorator to simplify the "both", "messages", "book" logic that is common to a few methods
    def add_ticker_column(
        self, to: t.Literal["both", "messages", "book"] = "messages"
    ) -> t.Self:
        if to in ("both", "messages"):
            self.messages = self.messages.assign(ticker=self.data.ticker).astype(
                {"ticker": "category"}
            )
        if to in ("both", "book"):
            self.book = self.book.assign(ticker=self.data.ticker).astype(
                {"ticker": "category"}
            )
        return self

    def __repr__(self) -> str:
        return f"Lobster data for ticker: {self.data.ticker} for date range: {self.data.date_range[0]} to {self.data.date_range[1]}."

# %% ../notebooks/01_preprocessing.ipynb 21
# | code-fold: true
class MPLobster:
    "Lobster data class for a single symbol of Lobster data."

    MAX_WORKERS: int = 70

    @staticmethod
    def process_messages(date, filepath):
        print(f"start {date}")
        df = pd.read_csv(
            filepath,
            header=None,
            nrows=None,
            usecols=list(range(6)),
            names=["time", "event", "order_id", "size", "price", "direction"],
            index_col=False,
            dtype={
                "time": "float64",
                "event": "int8",
                "price": "int64",
                "direction": "int8",
                "order_id": "int32",
                "size": "int64",
            },
        )

        # set index as datetime
        df["datetime"] = pd.to_datetime(date, format="%Y-%m-%d") + df.time.apply(
            lambda x: pd.to_timedelta(x, unit="s")
        )
        df.set_index("datetime", drop=True, inplace=True)
        print(f"finished {date}")
        return df

    @staticmethod
    def process_all_messages(messages_paths, max_workers=70):
        print(max_workers)
        mp.set_start_method("spawn")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            dfs = list(
                executor.map(
                    MPLobster.process_messages,
                    messages_paths.keys(),
                    messages_paths.values(),
                )
            )
            # dfs = (self.process_messages(date=date, filepath=filepath) for date, filepath in self.data.messages_paths.items())
        df = pd.concat(dfs)
        del dfs
        gc.collect()

        # direction for cross trades is set to zero, and order_id is left unchanged
        if not df.loc[df.event.eq(Event.CROSS_TRADE.value), "direction"].eq(-1).all():
            raise ValueError("All cross trades must have direction -1")
        df.loc[df.event.eq(Event.CROSS_TRADE.value), "direction"] = 0

        if (
            not df.loc[df.event.eq(Event.ORIGINAL_TRADING_HALT.value), "direction"]
            .eq(-1)
            .all()
        ):
            raise ValueError("All trading halts must have direction -1")
        df.loc[df.event.eq(Event.ORIGINAL_TRADING_HALT.value), "direction"] = 0

        # process trading halts and map to new trading halts
        def _trading_halt_type(price):
            return {
                -1: Event.TRADING_HALT.value,
                0: Event.RESUME_QUOTE.value,
                1: Event.TRADING_RESUME.value,
            }[price]

        df.loc[df.event.eq(Event.ORIGINAL_TRADING_HALT.value), "event"] = df.loc[
            df.event.eq(Event.ORIGINAL_TRADING_HALT.value), "price"
        ].apply(_trading_halt_type)

        # use 0 as NaN for size and direction
        df.loc[
            df.event.isin(EventGroup.HALTS.value),
            ["order_id", "size", "price"],
        ] = [0, 0, np.nan]

        # set price in dollars
        df["price"] = (df.price / 10_000).astype("float")
        return df

    @staticmethod
    def process_books(filename):
        print(f"start {filename}")
        col_names = []
        for level in range(1, 10 + 1):
            for col_type in ["ask_price", "ask_size", "bid_price", "bid_size"]:
                col_name = f"{col_type}_{level}"
                col_names.append(col_name)

        df = pd.read_csv(
            filename,
            header=None,
            nrows=None,
            usecols=None,
            names=col_names,
            dtype="float64",
            na_values=["-9999999999", "9999999999", "0"],
        )
        print(f"end {filename}")
        return df

    @staticmethod
    def process_all_books(book_paths: dict, max_workers=70):
        # check if set
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
        print(max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            dfs = list(executor.map(MPLobster.process_books, book_paths.values()))
        df = pd.concat(dfs)
        del dfs

        # for now just don't grab message index..
        # df.set_index(self.messages.index, inplace=True, drop=True)

        price_cols = df.columns.str.contains("price")
        df.loc[:, price_cols] = df.loc[:, price_cols] / 10_000
        return df

    def __init__(self, data):
        self.data = data

        if self.data.load_messages:
            self.messages = MPLobster.process_all_messages(
                messages_paths=self.data.messages_paths
            )
            gc.collect()

        if self.data.load_book:
            book = MPLobster.process_all_books(book_paths=self.data.book_paths)
            # set index here so that process_all_books can be a static_method
            book.set_index(self.messages.index, inplace=True, drop=True)
            self.book = book

        # everything below here not that important
        # TODO: think if leaving bool flags good idea
        if self.data.aggregate_duplicates:
            self.aggregate_duplicates()
        if self.data.clip_trading_hours:
            self.clip_trading_hours()
        if self.data.add_ticker_column:
            self.add_ticker_column()

    def clip_trading_hours(self) -> t.Self:
        if hasattr(self, "book"):
            self.book = _clip_to_trading_hours(self.book)
        if hasattr(self, "messages"):
            self.messages = _clip_to_trading_hours(self.messages)
        return self

    def aggregate_duplicates(self) -> t.Self:
        self.messages = _aggregate_duplicates(self.messages)
        return self

    # TODO: write decorator to simplify the "both", "messages", "book" logic that is common to a few methods
    def add_ticker_column(
        self, to: t.Literal["both", "messages", "book"] = "messages"
    ) -> t.Self:
        if to in ("both", "messages"):
            self.messages = self.messages.assign(ticker=self.data.ticker).astype(
                {"ticker": "category"}
            )
        if to in ("both", "book"):
            self.book = self.book.assign(ticker=self.data.ticker).astype(
                {"ticker": "category"}
            )
        return self

    def __repr__(self) -> str:
        return f"Lobster data for ticker: {self.data.ticker} for date range: {self.data.date_range[0]} to {self.data.date_range[1]}."

# %% ../notebooks/01_preprocessing.ipynb 26
@dataclass
class FolderInfo:
    full: str
    ticker: str
    ticker_till_end: str
    start_date: str
    end_date: str
    date_range: tuple[str, str] = field(init=False)

    def __post_init__(self):
        self.date_range = (self.start_date, self.end_date)


def infer_ticker_dict(
    files_path: str = "/nfs/home/nicolasp/home/data/tmp", /
) -> list[FolderInfo]:
    """Infer from folder structure the ticker to date_range mapping."""
    files_path_ = Path(files_path)
    all_folders = [str(path) for path in files_path_.glob("*")]

    ticker_pattern = re.compile(
        r"(.*?)(([A-Z.]{1,7})_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})(.*))"
    )

    folder_info = [
        FolderInfo(
            full=match.group(0),
            ticker=match.group(3),
            ticker_till_end=match.group(2).rstrip(".7z"),
            start_date=match.group(4),
            end_date=match.group(5),
        )
        for folder in all_folders
        if (match := ticker_pattern.search(folder))
    ]
    return folder_info
