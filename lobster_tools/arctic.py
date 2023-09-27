# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/07_arctic.ipynb.

# %% auto 0
__all__ = ['cfg', 'CONTEXT_SETTINGS', 'O', 'click_db_path', 'click_library', 'Options', 'apply_options', 'arctic', 'initdb',
           'use_both', 'dropdb', 'cool', 'nic', 'nic_entrypoint', 'get_arctic_library', 'inherit_docstring_from',
           'infer_options', 'inherits_from', 'list_libraries', 'list_symbols', 'create_library', 'delete_library',
           'read', 'write', 'say', 'generate_jobs', 'sleepy', 'extract_7z', 'zip', 'dump', 'arctic_list_symbols',
           'arctic_create_new_library', 'arctic_list_libraries', 'arctic_delete_library', 'arctic_read_symbol',
           'arctic_write_symbol', 'arctic_generate_jobs', 'zip_generate_jobs', 'arctic_dump_all']

# %% ../notebooks/07_arctic.ipynb 4
import os

import click
from arcticdb import Arctic, LibraryOptions
import hydra
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
from pathlib import Path
from lobster_tools.config import (
    MainConfig,
    Overrides,
    NASDAQExchange,
    ETFMembers,
    register_configs,
    get_config,
)
from .preprocessing import Data, Lobster, infer_ticker_to_date_range, infer_ticker_to_ticker_path, infer_ticker_dict
import sys
import pandas as pd
from logging import Logger
from datetime import date
from typing import Callable
from dataclasses import dataclass
import time
from inspect import signature
from functools import wraps

from concurrent.futures import ProcessPoolExecutor, wait
import subprocess

# %% ../notebooks/07_arctic.ipynb 6
register_configs()
cfg = get_config(overrides=Overrides.full_server)

# %% ../notebooks/07_arctic.ipynb 7
CONTEXT_SETTINGS = dict(
    help_option_names=["-h", "--help"],
    token_normalize_func=lambda x: x.lower() if isinstance(x, str) else x,
    show_default=True,
)

# %% ../notebooks/07_arctic.ipynb 10
class Options:
    def __init__(self) -> None:
        self.db_path = click.option("-d", "--db_path", default=cfg.db.db_path, help="Database path")
        self.library = click.option("-l", "--library", default=cfg.db.db_path, help="Library name")

def apply_options(options: list):
    def decorator(f):
        for option in reversed(options):
            f = option(f)
        return f
    return decorator

@click.group()
def arctic():
    pass

O = Options()
@arctic.command()
@apply_options([O.db_path])
def initdb(db_path):
    print(f'Initialized the database {db_path}')
    # click.echo(f'Initialized the database {db_path}')

@arctic.command()
@apply_options([O.db_path, O.library])
def use_both(db_path, library):
    print(f'Initialized the database {db_path} {library}')
    # click.echo(f'Initialized the database {db_path}')

@arctic.command()
def dropdb():
    click.echo('Dropped the database')

# %% ../notebooks/07_arctic.ipynb 11
click_db_path = click.option("-d", "--db_path", default=cfg.db.db_path, help="database path")
click_library = click.option("-l", "--library", default=cfg.db.library, help="library name")

@click.group()
def cool():
    pass

@cool.command()
@click_db_path
def initdb(db_path):
    print(f'Initialized the database {db_path}')
    # click.echo(f'Initialized the database {db_path}')

@cool.command()
def dropdb():
    click.echo('Dropped the database')

# %% ../notebooks/07_arctic.ipynb 12
@click.group()
@click.option("-d", "--db_path", default=cfg.db.db_path, help="database path")
@click.option("-l", "--library", default=cfg.db.library, help="library name")
@click.pass_context
def nic(ctx, db_path, library):
    ctx.ensure_object(dict)
    ctx.obj['DB_PATH'] = db_path
    ctx.obj['LIBRARY'] = library

@nic.command()
@click.pass_context
def initdb(ctx):
    db_path = ctx.obj['DB_PATH']
    print(f'Initialized the database {db_path}')
    click.echo(f'Initialized the database {db_path}')

@nic.command()
@click.pass_context
def dropdb(ctx):
    click.echo('Dropped the database')

def nic_entrypoint():
    nic(obj={}, auto_envvar_prefix='NIC')

# %% ../notebooks/07_arctic.ipynb 13
def get_arctic_library(db_path, library):
    conn = f"lmdb://{db_path}"
    arctic = Arctic(conn)
    arctic_library = arctic[library]
    return arctic_library

# %% ../notebooks/07_arctic.ipynb 15
# TODO: csv_path vs csv_files_path. think if this is a problem...maybe unify?
class Options:
    def __init__(self) -> None:
        self.db_path = click.option("-d", "--db_path", default=cfg.db.db_path, help="Database path")
        self.library = click.option("-l", "--library", default=cfg.db.library, help="Library name")
        self.ticker = click.option("-t", "--ticker", required=True, help="ticker to print")
        self.start_date = click.option("-s", "--start_date", default=None, help="start date")
        self.end_date = click.option("-e", "--end_date", default=None, help="end date")
        self.csv_path = click.option("-c", "--csv_path", default=cfg.data_config.csv_files_path, help="csv files path")
        self.etf = click.option("--etf", default=None, help="restrict to subset specified by ETF members")
        self.zip_path = click.option("-z", "--zip_path", default="/nfs/lobster_data/lobster_raw/2016", help="zip files path")
        self.tickers = click.option("--tickers", default=None, multiple=True, type=str, help="tickers to dump")
        self.max_workers = click.option("-m", "--max_workers", default=20, help="max workers for parallelisation")
O = Options()

def apply_options(options: list):
    def decorator(f):
        for option in reversed(options):
            f = option(f)
        return f
    return decorator

def inherit_docstring_from(source_fn):
    def decorator(target_fn):
        target_fn.__doc__ = source_fn.__doc__
        return target_fn
    return decorator

def infer_options(func) -> list[Callable]:
    """Works together with the `auto_apply` to automatically infer arguments.
    
    Used together this looks like:
    @auto_apply(infer_options)
    """
    sig = signature(func)
    param_names = [
        param.name
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    ]
    options_list = [getattr(O, name) for name in param_names]
    return options_list

def inherits_from(func):
    """Inherit docstring and options from `func`."""
    options_list = infer_options(func)

    # still stlightly confused about the order of the decorators, oh well
    def decorator(target_fn):
        @inherit_docstring_from(func)
        @apply_options(options_list)
        @wraps(target_fn)
        def wrapper(*args, **kwargs):
            return target_fn(*args, **kwargs)
        return wrapper
    return decorator

# def simple_inherits_from(func):
#     """Simple without using functools.wraps"""
#     options_list = infer_options(func)
#     def decorator(target_fn):
#         decorated = apply_options(options_list)(target_fn)
#         decorated.__doc__ = func.__doc__
#         return decorated
#     return decorator

@click.group(context_settings=CONTEXT_SETTINGS)
def arctic():
    pass

@arctic.command()
@apply_options([O.db_path])
def list_libraries(db_path) -> None:
    """List arcticdb libraries"""
    arctic = Arctic(f"lmdb://{db_path}")
    print(arctic.list_libraries())

@arctic.command()
@apply_options([O.db_path, O.library])
def list_symbols(db_path, library) -> None:
    """List symbols in the arcticdb library."""
    arctic = Arctic(f"lmdb://{db_path}")
    print(arctic[library].list_symbols())

@arctic.command()
@apply_options([O.db_path, O.library])
def create_library(db_path, library) -> None:
    """Create a blank new arcticdb library."""
    arctic = Arctic(f"lmdb://{db_path}")
    arctic.create_library(library) 
    print(arctic[library])

@arctic.command()
@apply_options([O.db_path, O.library])
@click.confirmation_option(prompt='Are you sure you want to delete the entire library?')
def delete_library(db_path, library) -> None:
    """Delete entire arcticdb library"""
    arctic = Arctic(f"lmdb://{db_path}")
    arctic.delete_library(library) 

@arctic.command()
@apply_options([O.db_path, O.library, O.ticker, O.start_date, O.end_date])
def read(db_path, library, ticker, start_date, end_date,
):
    """Read ticker and print head and tail."""
    arctic = Arctic(f"lmdb://{db_path}")

    if start_date and end_date:
        start_datetime = pd.Timestamp(f"{start_date}T{NASDAQExchange.exchange_open}")
        end_datetime = pd.Timestamp(f"{end_date}T{NASDAQExchange.exchange_close}")
        date_range = (start_datetime, end_datetime)
        df = arctic[library].read(ticker, date_range=date_range).data
    else:
        print("not using start or end dates")
        df = arctic[library].read(ticker).data
    
    print(f"Printing df.head() and df.tail() for ticker {ticker}")
    print(df.head())
    print(df.tail())

def _write(
    db_path,
    library,
    csv_path,
    ticker,
    start_date,
    end_date,
):
    """Preprocess and write ticker to database."""
    arctic = Arctic(f"lmdb://{db_path}")

    date_range = (start_date, end_date)
    data = Data(
        directory_path=csv_path,
        ticker=ticker,
        date_range=date_range,
        aggregate_duplicates=False,
    )
    lobster = Lobster(data=data)
    df = pd.concat([lobster.messages, lobster.book], axis=1)
    print(f"head of ticker {ticker}")
    print(df.head())

    arctic[library].write(symbol=ticker, data=df)

@arctic.command()
@apply_options([O.db_path, O.library, O.csv_path, O.ticker, O.start_date, O.end_date])
def write(**kwargs):
    _write(**kwargs)


# if want to also access _say from other functions then need to do this.
def _say(
    db_path,
    library,
):
    """Print some really important information"""
    print(db_path, library)


# @arctic.command()
# @apply_options(infer_options(_say))
# @inherit_docstring_from(_say)
# def say(**kwargs):
#     _say(**kwargs)

@arctic.command()
@inherits_from(_say)
def say(**kwargs):
    _say(**kwargs)

@arctic.command()
@apply_options([O.db_path, O.library, O.csv_path, O.start_date, O.end_date])
def generate_jobs(db_path, library, csv_path, start_date, end_date):
    ticker_date_dict = infer_ticker_to_date_range(csv_path)
    with open('arctic_commands.txt', 'w') as f:
        for ticker, (inferred_start_date, inferred_end_date) in ticker_date_dict.items():
            # if date is None use the inferred date, otherwise use the CLI argument
            start_date = start_date or inferred_start_date
            end_date = end_date or inferred_end_date
            f.write(f"arctic write --csv_path={csv_path} --db_path={db_path} --library={library} --ticker={ticker} --start_date={start_date} --end_date={end_date} \n")

def sleepy(csv_path, folder_info):
    time.sleep(5)
    print(csv_path, folder_info.full)

def extract_7z(input_path, output_path):
    try:
        subprocess.run(["7z", "x", input_path, f"-o{output_path}"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {str(e)}")

@arctic.command()
@apply_options([O.zip_path, O.csv_path, O.etf, O.max_workers])
def zip(zip_path, csv_path, etf, max_workers):
    folder_infos = infer_ticker_dict(zip_path)

    # filter first
    if etf:
        def in_etf(folder_info):
            return folder_info.ticker in ETFMembers().mapping[etf] + [etf]
        folder_infos = list(filter(in_etf, folder_infos))

    # commands = [f"mkdir -p {csv_path}/{folder_info.ticker_till_end}\n"
    #             for folder_info in folder_infos]

    # with open("zip_commands.txt", "w") as f:
    #     [f.write(command) for command in commands]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # outputs_dirs = [folder_info.ticker_till_end for folder_info in folder_infos]
        futures = [
            executor.submit(os.mkdir, path=f"{csv_path}/{folder_info.ticker_till_end}")
            for folder_info in folder_infos
        ]
        wait(futures)
        futures = [
            executor.submit(extract_7z, input_path=folder_info.full, output_path=f"{csv_path}/{folder_info.ticker_till_end}")
            for folder_info in folder_infos
        ]


        # for folder_info in folder_infos:
        #     # print(folder_info.ticker)
        #     f.write(f"examle: mkdir {csv_path}/{folder_info.ticker_till_end}\n")


    # ticker_date_dict = infer_ticker_to_ticker_path(zip_path)
    # print(ticker_date_dict)
    # if etf:
    #     print(ETFMembers().mapping[etf])
    #     ticker_date_dict = {
    #         ticker: ticker_path
    #         for ticker, ticker_path in ticker_date_dict.items()
    #         if ticker in ETFMembers().mapping[etf] + [etf]
    #     }
    # print(ticker_date_dict)
    # ticker_dict = infer_ticker_dict(zip_path)
    # with open("zip_commands.txt", "w") as f:
    #     for ticker, dict_ in ticker_dict.items():
    #         full = dict_["full"]
    #         ticker_till_end = dict_["ticker_till_end"]
    #         f.write(f"mkdir {csv_path}/{ticker_till_end}\n")
    #         f.write(f"/nfs/home/nicolasp/usr/bin/7z x {full} -o{ticker_till_end}\n")


@arctic.command()
@apply_options([O.db_path, O.library, O.csv_path, O.tickers, O.max_workers])
def dump(
    db_path,
    library,
    csv_path,
    tickers,
    max_workers,
):
    """Dump all csv to arctic_db inferring start and end date from folder."""
    folder_infos = infer_ticker_dict(csv_path)
    print("inferred from folder")
    print(folder_infos)

    if tickers:
        folder_infos = [folder_info for folder_info in folder_infos if folder_info.ticker in tickers]

    print("filtered folder_info after filtering for tickers.")
    print(folder_infos)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # small job with only a few dates
        # futures = [
        #     executor.submit(write_, csv_path=csv_path, db_path=db_path, library=library, ticker=folder_info.ticker, start_date="2016-01-01", end_date="2016-01-04")
        #     for folder_info in folder_infos
        # ]
        # full job with whole year
        futures = [
            executor.submit(_write, csv_path=csv_path, db_path=db_path, library=library, ticker=folder_info.ticker, start_date=folder_info.start_date, end_date=folder_info.end_date)
            for folder_info in folder_infos
        ]
    print('done')

# %% ../notebooks/07_arctic.ipynb 18
# | code-fold: true
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-d", "--db_path", default=cfg.db.db_path, help="database path")
@click.option("-l", "--library", default=cfg.db.library, help="library name")
def arctic_list_symbols(db_path, library) -> None:
    """List symbols in the arcticdb library."""
    arctic_library = get_arctic_library(db_path=db_path, library=library)
    print(f"Symbols in library {library}")
    print(arctic_library.list_symbols())

# %% ../notebooks/07_arctic.ipynb 19
# | code-fold: true
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-d", "--db_path", default=cfg.db.db_path, help="database path")
@click.option("-l", "--library", default=cfg.db.library, help="library name")
def arctic_create_new_library(db_path, library) -> None:
    """Create a blank new arcticdb library."""
    conn = f"lmdb://{db_path}"
    arctic = Arctic(conn)
    arctic.create_library(library) 
    print(arctic[library])

# %% ../notebooks/07_arctic.ipynb 20
# | code-fold: true
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-d", "--db_path", default=cfg.db.db_path, help="database path")
def arctic_list_libraries(db_path) -> None:
    """List arcticdb libraries"""

    conn = f"lmdb://{db_path}"
    arctic = Arctic(conn)
    print(arctic.list_libraries())

# %% ../notebooks/07_arctic.ipynb 21
# | code-fold: true
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-d", "--db_path", default=cfg.db.db_path, help="database path")
@click.option("-l", "--library", default=cfg.db.library, help="library name")
def arctic_delete_library(db_path, library) -> None:
    """Delete arcticdb library"""

    user_input = input("Proceed by deleting this entire library? (y/n): ")
    user_input = user_input.lower()
    match user_input:
        case "y":
            pass
        case "n":
            sys.exit(0)
        case _:
            sys.exit(1)

    conn = f"lmdb://{db_path}"
    arctic = Arctic(conn)
    arctic.delete_library(library) 

# %% ../notebooks/07_arctic.ipynb 22
# | code-fold: true
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-d", "--db_path", default=cfg.db.db_path, help="database path")
@click.option("-l", "--library", default=cfg.db.library, help="library name")
@click.option("-t", "--ticker", required=True, help="ticker to print")
@click.option("-s", "--start_date", default=None, help="start date")
@click.option("-e", "--end_date", default=None, help="end date")
def arctic_read_symbol(db_path, library, ticker, start_date, end_date,
):
    """Print df.head() and available columns for ticker in arcticdb library."""
    arctic_library = get_arctic_library(db_path=db_path, library=library)

    if start_date and end_date:
        start_datetime = pd.Timestamp(f"{start_date}T{NASDAQExchange.exchange_open}")
        end_datetime = pd.Timestamp(f"{end_date}T{NASDAQExchange.exchange_close}")
        date_range = (start_datetime, end_datetime)
        df = arctic_library.read(ticker, date_range=date_range).data
    else:
        df = arctic_library.read(ticker).data
    
    print(f"Printing df.head() and df.tail() for ticker {ticker}")
    print(df.head())
    print(df.tail())

# %% ../notebooks/07_arctic.ipynb 24
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

    # if ticker in arctic_library.list_symbols():
    #     print("warning - there is already data for ths ticker")
    #     user_input = input("Proceed by adding data to this symbol? (y/n): ")
    #     user_input = user_input.lower()
    #     match user_input:
    #         case "y":
    #             pass
    #         case "n":
    #             sys.exit(0)
    #         case _:
    #             sys.exit(1)

    date_range = (start_date, end_date)
    data = Data(
        directory_path=csv_path,
        ticker=ticker,
        date_range=date_range,
        aggregate_duplicates=False,
    )
    lobster = Lobster(data=data)
    df = pd.concat([lobster.messages, lobster.book], axis=1)

    arctic_library.write(symbol=ticker, data=df)

# %% ../notebooks/07_arctic.ipynb 25
# | code-fold: true
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-c", "--csv_path", default=cfg.data_config.csv_files_path, help="csv files path"
)
@click.option("-d", "--db_path", default=cfg.db.db_path, help="database path")
@click.option("-l", "--library", default=cfg.db.library, help="library name")
@click.option("-s", "--start_date", default=None, help="start date")
@click.option("-e", "--end_date", default=None, help="end date")
def arctic_generate_jobs(csv_path, db_path, library, start_date, end_date):
    ticker_date_dict = infer_ticker_to_date_range(csv_path)
    with open('arctic_commands.txt', 'w') as f:
        for ticker, (inferred_start_date, inferred_end_date) in ticker_date_dict.items():
            # if date is None use the inferred date, otherwise use the CLI argument
            start_date = start_date or inferred_start_date
            end_date = end_date or inferred_end_date
            f.write(f"arctic_write_symbol --csv_path={csv_path} --db_path={db_path} --library={library} --ticker={ticker} --start_date={start_date} --end_date={end_date} \n")

# %% ../notebooks/07_arctic.ipynb 26
# | code-fold: true
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-z",
    "--zip_path",
    default="/nfs/lobster_data/lobster_raw/2016",
    help="zip files path",
)
@click.option(
    "-c", "--csv_path", default=cfg.data_config.csv_files_path, help="csv files path"
)
@click.option(
    "-e", "--etf", default=None, help="restrict to subset specified by ETF members"
)
def zip_generate_jobs(zip_path, csv_path, etf):
    # ticker_date_dict = infer_ticker_to_ticker_path(zip_path)
    # print(ticker_date_dict)
    # if etf:
    #     print(ETFMembers().mapping[etf])
    #     ticker_date_dict = {
    #         ticker: ticker_path
    #         for ticker, ticker_path in ticker_date_dict.items()
    #         if ticker in ETFMembers().mapping[etf] + [etf]
    #     }
    # print(ticker_date_dict)
    ticker_dict = infer_ticker_dict(zip_path)
    with open("zip_commands.txt", "w") as f:
        for ticker, dict_ in ticker_dict.items():
            full = dict_["full"]
            ticker_till_end = dict_["ticker_till_end"]
            f.write(f"mkdir {csv_path}/{ticker_till_end}\n")
            f.write(f"/nfs/home/nicolasp/usr/bin/7z x {full} -o{ticker_till_end}\n")

# %% ../notebooks/07_arctic.ipynb 27
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
def arctic_dump_all(
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
        user_input = input("Proceed by adding data to this symbol? (y/n): ")
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
