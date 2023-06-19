# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/01_querying.ipynb.

# %% auto 0
__all__ = ['get_buy', 'get_sell', 'get_executions', 'get_halts', 'get_cancellations', 'load_equity_executions',
           'load_etf_executions', 'get_equity', 'get_etf', 'query_by_direction', 'split_by_direction', 'query_by_event',
           'load_executions', 'query_ticker_type']

# %% ../notebooks/01_querying.ipynb 4
from .preprocessing import *
import pandas as pd
from functools import partial
import datetime


# %% ../notebooks/01_querying.ipynb 5
def query_by_direction(
    df: pd.DataFrame,  # messages dataframe
    direction: str | Direction,  # direction, either "buy" or "sell"
) -> pd.DataFrame:
    """Query a messages dataframe on the direction column."""
    if isinstance(direction, Direction):
        direction = direction.value
    elif isinstance(direction, str):
        direction = direction.lower()
        direction_str_to_int = {"buy": 1, "sell": -1}
        if direction not in direction_str_to_int:
            raise ValueError(f"{direction} is not a valid direction.")
        direction = direction_str_to_int[direction]
    return df.query("direction == @direction")


get_buy = partial(query_by_direction, direction="buy")
get_sell = partial(query_by_direction, direction="sell")


def split_by_direction(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    "Returns a tuple of (buy, sell) DataFrames"
    return get_buy(df), get_sell(df)


# %% ../notebooks/01_querying.ipynb 6
def query_by_event(
    df: pd.DataFrame,  # messages dataframe
    event: str | Event | EventGroup,  # event as str or `Event` or `EventGroup`.
) -> pd.DataFrame:
    """Query a messages dataframe on the event column."""
    if isinstance(event, str):
        event_str_to_enum = {
            "unknown": Event.UNKNOWN.value,
            "submission": Event.SUBMISSION.value,
            "cancellation": Event.CANCELLATION.value,
            "deletion": Event.DELETION.value,
            "execution": Event.EXECUTION.value,
            "hidden execution": Event.HIDDEN_EXECUTION.value,
            "cross trade": Event.CROSS_TRADE.value,
            "trading halt": Event.TRADING_HALT.value,
            "other": Event.OTHER.value,
            "resume quote": Event.RESUME_QUOTE.value,
            "trading resume": Event.TRADING_RESUME.value,
            "executions": EventGroup.EXECUTIONS.value,
            "halts": EventGroup.HALTS.value,
            "cancellations": EventGroup.CANCELLATIONS.value,
        }
        event = event.lower().replace("_", " ")

        if event not in event_str_to_enum:
            raise ValueError(f"event must be one of {list(event_str_to_enum.keys())}")

        event = event_str_to_enum[event]

    elif isinstance(event, Event | EventGroup):
        event = event.value

    if isinstance(event, list):
        return df.query("event in @event")
    elif isinstance(event, int):
        return df.query("event == @event")

# %% ../notebooks/01_querying.ipynb 7
get_executions = partial(query_by_event, event="executions")
get_halts = partial(query_by_event, event="halts")
get_cancellations = partial(query_by_event, event="cancellations")


def load_executions(date_range: str, tickers: list[str], ticker_type: str) -> pd.DataFrame:
    ticker_execution_dfs = []
    for ticker in tickers:
        ticker_execution_dfs.append(
            load_lobster(
                ticker=ticker,
                date_range=date_range,
                lobster_only=True,
                add_ticker=True,
                ticker_type=ticker_type,
            ).messages.pipe(get_executions)
        )

    tickers_execution = pd.concat(ticker_execution_dfs).astype(dtype={"ticker": "category"})
    return tickers_execution


load_equity_executions = partial(load_executions, ticker_type="equity")
load_etf_executions = partial(load_executions, ticker_type="etf")

# %% ../notebooks/01_querying.ipynb 8
def query_ticker_type(df, ticker_type):
    return df.query("ticker_type == @ticker_type")


get_equity = partial(query_ticker_type, ticker_type="equity")
get_etf = partial(query_ticker_type, ticker_type="etf")
