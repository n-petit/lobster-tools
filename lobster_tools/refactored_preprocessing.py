import datetime
import enum
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from absl import app, flags
import ray

import lobster_tools.config  # noqa: F401

FLAGS = flags.FLAGS


@enum.unique
class Event(enum.Enum):
    "Event types for LOBSTER messages."
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
    AGGREGATED = 12


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

@dataclass
class FileMetadataFromCSV:
    csv_path: str
    ticker: str
    date: str
    orderbook_or_message: Literal["orderbook", "message"]
    levels: int

    # TODO: i guess could put all tests here or just in a separate test file..
    def __post_init__(self):
        assert self.orderbook_or_message in ("orderbook", "message")
        assert re.match(r"\d{4}-\d{2}-\d{2}", self.date)


def _filter_by_date(
    files: list[FileMetadataFromCSV], date_range: tuple[str, str]
) -> list[FileMetadataFromCSV]:
    start_date, end_date = date_range
    # str comparisons on YYYY-MM-DD format are valid
    filtered_files = [f for f in files if start_date <= f.date <= end_date]
    return filtered_files


class OrderbooksAndMessages(NamedTuple):
    orderbooks: list[FileMetadataFromCSV]
    messages: list[FileMetadataFromCSV]


def split_into_orderbooks_and_messages(
    files: list[FileMetadataFromCSV],
) -> OrderbooksAndMessages:
    return OrderbooksAndMessages(
        orderbooks=[f for f in files if f.orderbook_or_message == "orderbook"],
        messages=[f for f in files if f.orderbook_or_message == "message"],
    )


def infer_metadata_from_csv_files(
    dir_path: str = "/nfs/home/nicolasp/home/data/tmp/KO",
    *,
    date_range: tuple[str, str] | None = None,
) -> OrderbooksAndMessages:
    """Infer ticker name, date, book_or_message and levels from regex pattern matching.
    Typical files in directory would be:

    .
    ├── AAPL-01-04_34200000_57600000_message_10.csv
    ├── AAPL-01-04_34200000_57600000_orderbook_10.csv
    └── AAPL-09-02_34200000_57600000_message_10.csv
    """
    dir_contents = Path(dir_path).glob("*")
    dir_contents = map(str, dir_contents)

    ticker_name_pattern = r"[A-Z.]{1,6}"
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    orderbook_or_message_pattern = r"orderbook|message"
    level_pattern = r"\d{1,2}"

    full_pattern = re.compile(
        rf"((.*?)({ticker_name_pattern})_({date_pattern})_(.*)_({orderbook_or_message_pattern})_({level_pattern})(.*))"
    )

    files = [
        FileMetadataFromCSV(
            csv_path=match.group(0),
            ticker=match.group(3),
            date=match.group(4),
            orderbook_or_message=match.group(6),
            levels=int(match.group(7)),
        )
        for file in dir_contents
        if (match := full_pattern.search(file))
    ]
    assert files, "No files found in directory"

    files = sorted(files, key=lambda x: x.date)

    if date_range:
        files = _filter_by_date(files, date_range)

    files = split_into_orderbooks_and_messages(files)

    return files


def load_messages(files: list[FileMetadataFromCSV]):
    check(files, "message")

    cols = ("time", "event", "order_id", "size", "price", "direction")
    col_dtypes = {
        "time": "float64",
        "event": "int8",
        "order_id": "int32",
        "size": "int64",
        "price": "int64",
        "direction": "int8",
    }

    @ray.remote
    def load_single_message(path: str, date: str):
        df = pd.read_csv(
            path,
            header=None,
            names=cols,
            usecols=list(range(len(cols))),
            index_col=False,
            dtype=col_dtypes,
        )

        df["datetime"] = pd.to_datetime(date, format="%Y-%m-%d") + pd.to_timedelta(
            df.time, unit="s"
        )

        df.set_index("datetime", drop=True, inplace=True)

        df.price = df.price / 10_000  # convert price to dollars

        df = process_cross_trades(df)
        df = process_trading_halts(df)

        return df

    dfs = ray.get([load_single_message.remote(path=f.csv_path, date=f.date) for f in files])
    df = pd.concat(dfs)

    assert df.index.is_monotonic_increasing, "Index is not sorted."

    return df


def process_trading_halts(df: pd.DataFrame) -> pd.DataFrame:
    "Replace trading halt direction, order_id, size and price fields."
    assert (
        df.loc[df.event.eq(Event.ORIGINAL_TRADING_HALT.value), "direction"].eq(-1).all()
    ), "All trading halts must have direction -1"
    df.loc[df.event.eq(Event.ORIGINAL_TRADING_HALT.value), "direction"] = 0

    trading_halt_mask = df.event.eq(Event.ORIGINAL_TRADING_HALT.value)
    halt_type_mapping = {
        -1: Event.TRADING_HALT.value,
        0: Event.RESUME_QUOTE.value,
        1: Event.TRADING_RESUME.value,
    }

    df.loc[trading_halt_mask, "event"] = (
        df.loc[trading_halt_mask, "price"]
        .map(halt_type_mapping)
        .astype(df["event"].dtype)
    )

    df.loc[
        df.event.isin(EventGroup.HALTS.value),
        ["order_id", "size", "price"],
    ] = [0, 0, np.nan]

    return df


def process_cross_trades(df: pd.DataFrame) -> None:
    "Set cross trade direction field to zero."
    assert (
        df.loc[df.event.eq(Event.CROSS_TRADE.value), "direction"].eq(-1).all()
    ), "Cross trades must have direction -1"
    df.loc[df.event.eq(Event.CROSS_TRADE.value), "direction"] = 0

    return df


def check(
    files: list[FileMetadataFromCSV],
    orderbook_or_message: Literal["orderbook", "message"],
) -> None:
    assert files == sorted(files, key=lambda f: f.date), "Dates not sorted."
    assert all(
        file_metadata.levels == files[0].levels for file_metadata in files
    ), "Levels are not the same in all files."
    assert all(
        f.orderbook_or_message == orderbook_or_message for f in files
    ), f"Not all files are {orderbook_or_message} files."


def load_orderbooks(files: list[FileMetadataFromCSV]) -> pd.DataFrame:
    check(files, "orderbook")

    levels = files[0].levels
    col_names = [
        f"{col_type}_{level}"
        for level in range(1, levels + 1)
        for col_type in ("ask_price", "ask_size", "bid_price", "bid_size")
    ]
    price_cols = [col for col in col_names if "price" in col]

    @ray.remote
    def load_single_orderbook(path: str):
        df = pd.read_csv(
            path,
            header=None,
            usecols=list(range(4 * levels)),
            names=col_names,
            dtype="int",  # integers in raw csv files
            na_values=["-9999999999", "9999999999", "0"],
        )

        df.loc[:, price_cols] = (
            df.loc[:, price_cols] / 10_000
        )  # convert price to dollars

        return df

    dfs = ray.get([load_single_orderbook.remote(path=f.csv_path) for f in files])

    df = pd.concat(dfs)

    return df

###################################################
# INFER FOLDER INFORMATION
###################################################

@dataclass
class MetadataFrom7zFile:
    zip_path: str
    ticker: str
    ticker_till_end: str
    start_date: str
    end_date: str
    date_range: tuple[str, str] = field(init=False)

    def __post_init__(self):
        self.date_range = (self.start_date, self.end_date)


def infer_metadata_from_7z_files(
    dir_path: str = "/nfs/data/lobster_data/lobster_raw/2021/", /
) -> list[MetadataFrom7zFile]:
    """Infer ticker name and date range from regex pattern matching. Typical directory 
    structure would be the following:

    .
    ├── _data_dwn_32_302__AAPL_2021-01-01_2021-12-31_10.7z
    ├── _data_dwn_32_302__MSFT_2021-01-01_2021-12-31_10.7z
    └── _data_dwn_32_302__SPY_2021-01-01_2021-12-31_10.7z
    """
    dir_contents = Path(dir_path).glob("*")
    dir_contents = map(str, dir_contents)

    ticker_name_pattern = r"[A-Z.]{1,6}"
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    full_pattern = re.compile(
        rf"(.*?)(({ticker_name_pattern})_({date_pattern})_({date_pattern})(.*))"
    )

    files = [
        MetadataFrom7zFile(
            zip_path=match.group(0),
            ticker=match.group(3),
            ticker_till_end=match.group(2).rstrip(".7z"),
            start_date=match.group(4),
            end_date=match.group(5),
        )
        for folder in dir_contents
        if (match := full_pattern.search(folder))
    ]
    return files

def clip_to_nasdaq_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    open, close = datetime.time(9, 30), datetime.time(16, 0)
    return df.between_time(open, close)


def main(_):
    ray.init()

    files = infer_metadata_from_csv_files(
        "/nfs/home/nicolasp/home/data/tmp/KO", date_range=("2021-01-04", "2021-01-05")
    )

    print("================orderbook=============")
    print(files.orderbooks)
    print("================messages=============")
    print(files.messages)

    book = load_orderbooks(files.orderbooks)
    print(book.head())
    print(book.tail())

    messages = load_messages(files.messages)
    print(messages.head())
    print(messages.tail())

    ray.shutdown()


if __name__ == "__main__":
    app.run(main)
