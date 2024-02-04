from absl import flags
from absl import logging
from absl import app

import datetime as dt
import pandas as pd
import typing as t

from arcticdb import Arctic, QueryBuilder
from arcticdb.version_store.library import Library

flags.DEFINE_string(
    "s3_uri",
    "s3://10.146.104.101:9100:lobster?access=minioadmin&secret=minioadmin",
    "URI of the S3 bucket to connect to.",
)
flags.DEFINE_string(
    "library",
    "2021",
    "Arctic library to connect to.",
)

FLAGS = flags.FLAGS


def test_database_connection(arctic: Arctic) -> pd.DataFrame:
    "Test connection to s3 arcticdb database by connecting to the '2021' library and reading the symbol 'FLIR'."
    available_libraries = arctic.list_libraries()
    print(f"available libraries: {available_libraries}")

    ticker, library = "FLIR", "2021"
    assert library in available_libraries
    assert ticker in arctic[library]

    date_range = (dt.datetime(2021, 1, 1), dt.datetime(2021, 1, 8))
    df = arctic[library].read(symbol=ticker, date_range=date_range).data

    logging.info(f"data.head() for ticker={ticker}, date_range={date_range}")
    logging.info(df.head())

    return df


DateRange = tuple[dt.datetime, dt.datetime] | dt.datetime
def _fetch_ticker(
    ticker: str,
    arctic_library: Library,
    columns: t.Optional[list[str]] = None,
    query_builder: t.Optional[QueryBuilder] = None,
    date_range: t.Optional[DateRange] = None,
):
    logging.info(f"Fetching data for ticker={ticker}")
    df = arctic_library.read(
        symbol=ticker,
        query_builder=query_builder,
        columns=columns,
        date_range=date_range,
    ).data

    df = df.assign(ticker=ticker).astype({"ticker": "category"})
    return df

def assign_mid(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        mid=lambda _df: (_df["bid_price_1"] + _df["ask_price_1"]) / 2
    )

def fetch_tickers(
    tickers: list,
    arctic_library: Library,
    columns: t.Optional[list[str]] = None,
    query_builder: t.Optional[QueryBuilder] = None,
    date_range: t.Optional[DateRange] = None,
):
    df = pd.concat(
            (
                _fetch_ticker(
                    ticker=ticker,
                    arctic_library=arctic_library,
                    columns=columns,
                    query_builder=query_builder,
                    date_range=date_range,
                )
                for ticker in tickers
            )
        ).sort_index().astype({"ticker": "category"})
    return df


def main(_):
    arctic = Arctic(FLAGS.s3_uri)
    arctic_library = arctic[FLAGS.library]

    df = fetch_tickers(
        ["FLIR", "YUM"],
        arctic_library=arctic_library,
        date_range=(dt.datetime(2021, 1, 1), dt.datetime(2021, 1, 8)),
    )
    logging.info(df.head())


if __name__ == "__main__":
    app.run(main)
