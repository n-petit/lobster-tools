# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/07_arctic.ipynb.

# %% auto 0
__all__ = ['cfg', 'CONTEXT_SETTINGS', 'get_arctic_library', 'arctic_read_symbol', 'arctic_write_symbol']

# %% ../notebooks/07_arctic.ipynb 4
import os

import click
from arcticdb import Arctic, LibraryOptions
import hydra
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
from pathlib import Path
from .config import MainConfig, Overrides, NASDAQExchange, register_configs, get_config
from .preprocessing import Data, Lobster
import sys
import pandas as pd
from logging import Logger
from datetime import date

# %% ../notebooks/07_arctic.ipynb 8
# register_configs()
cfg = get_config(overrides=Overrides.full_server)
cfg = get_config(overrides=['data_config=server', 'hyperparameters=full', 'universe=SPY'])

# %% ../notebooks/07_arctic.ipynb 9
CONTEXT_SETTINGS = dict(
    help_option_names=["-h", "--help"],
    token_normalize_func=lambda x: x.lower() if isinstance(x, str) else x,
    show_default=True,
)

# %% ../notebooks/07_arctic.ipynb 10
def get_arctic_library(db_path, library):
    conn = f"lmdb://{db_path}"
    arctic = Arctic(conn)
    arctic_library = arctic[library]
    return arctic_library

# %% ../notebooks/07_arctic.ipynb 12
# | code-fold: true
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-d", "--db_path", default=cfg.db.db_path, help="database path")
@click.option("-l", "--library", default=cfg.db.library, help="library name")
@click.option("-t", "--ticker", required=True, help="ticker to print")
@click.option("-s", "--start_date", default="2020-01-02", help="start date")
@click.option("-e", "--end_date", default="2020-01-07", help="end date")
def arctic_read_symbol(db_path, library, ticker, start_date, end_date,
):
    """Print df.head() and available columns for ticker in arcticdb library."""
    arctic_library = get_arctic_library(db_path=db_path, library=library)

    start_datetime = pd.Timestamp(f"{start_date}T{NASDAQExchange.exchange_open}")
    end_datetime = pd.Timestamp(f"{end_date}T{NASDAQExchange.exchange_close}")
    date_range = (start_datetime, end_datetime)
    df = arctic_library.read(ticker, date_range=date_range).data
    
    print(f"Printing df.head() for ticker {ticker}")
    print(df.tail())

# %% ../notebooks/07_arctic.ipynb 14
# | code-fold: true
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-c", "--csv_path", default=cfg.data_config.csv_files_path, help="csv files path"
)
@click.option("-d", "--db_path", default=cfg.db.db_path, help="database path")
@click.option("-l", "--library", default=cfg.db.library, help="library name")
@click.option("-t", "--ticker", required=True, help="ticker to write to db")
@click.option("-s", "--start_date", default="2020-01-01", help="start date")
@click.option("-e", "--end_date", default="2020-02-01", help="end date")
def arctic_write_symbol(
    db_path,
    library,
    csv_path,
    ticker,
    start_date,
    end_date,
):
    arctic_library = get_arctic_library(db_path=db_path, library=library)

    if ticker in arctic_library.list_symbols():
        print("warning - there is already data for ths ticker")
        user_input = input("Proceed by adding data to this symbol? (y/n):")
        user_input = user_input.lower()
        match user_input:
            case "y":
                pass
            case "n":
                sys.exit(0)
            case _:
                sys.exit(1)

    date_range = (start_date, end_date)
    data = Data(
        directory_path=csv_path,
        ticker=ticker,
        date_range=date_range,
        aggregate_duplicates=False,
    )
    lobster = Lobster(data=data)
    df = pd.concat([lobster.messages, lobster.book], axis=1)
    print(df)

    arctic_library.append(symbol=ticker, data=df)
