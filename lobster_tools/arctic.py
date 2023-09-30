# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/07_arctic.ipynb.

# %% auto 0
__all__ = ['cfg', 'CONTEXT_SETTINGS', 'O', 'click_db_path', 'click_library', 'C', 'Options', 'apply_options', 'arctic', 'initdb',
           'use_both', 'dropdb', 'cool', 'nic', 'read', 'get_arctic_library', 'ArcticLibraryInfo', 'get_library_info',
           'ConsoleNotify', 'ClickCtxObj', 'ClickCtx', 'echo', 'init', 'create', 'ls', 'libraries', 'symbols',
           'versions', 'parse_comma_separated', 'dates', 'rm', 'library', 'etf', 'query', 'finfo', 'add', 'write',
           'arctic_list_symbols', 'arctic_create_new_library', 'arctic_list_libraries', 'arctic_delete_library',
           'arctic_read_symbol', 'arctic_write_symbol', 'arctic_generate_jobs', 'zip_generate_jobs', 'arctic_dump_all']

# %% ../notebooks/07_arctic.ipynb 4
import os

import re
import gc
import json
from string import Template
import click
from click.testing import CliRunner
from arcticdb import Arctic, QueryBuilder
from arcticdb.version_store.library import Library
from arcticdb.exceptions import LibraryNotFound
import hydra
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
from pathlib import Path
from pprint import pformat
import textwrap
from lobster_tools.config import (
    MainConfig,
    Overrides,
    NASDAQExchange,
    ETFMembers,
    etf_to_equities,
    register_configs,
    get_config,
)
from .preprocessing import Data, Lobster, Event, infer_ticker_to_date_range, infer_ticker_to_ticker_path, infer_ticker_dict
import sys
import pandas as pd
import numpy as np
import logging
from logging import Logger
from datetime import date
from typing import Callable, TypedDict, Protocol, NotRequired, Required, cast
from dataclasses import dataclass, asdict
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
    ctx.obj['library'] = library

    conn = f"lmdb://{db_path}"
    arctic = Arctic(conn)
    ctx.obj['arctic'] = arctic

@nic.command()
@click.option("-t", "--ticker", default="AMZN", help="ticker")
@click.pass_context
def read(ctx, ticker):
    arctic = ctx.obj["arctic"]
    library = ctx.obj["library"]
    df = arctic[library].read(ticker).data
    print(f'df.head() {df.head()}')

@nic.command()
@click.pass_context
def dropdb(ctx):
    click.echo('Dropped the database')

# %% ../notebooks/07_arctic.ipynb 13
def get_arctic_library(db_path, library):
    conn = f"lmdb://{db_path}"
    arctic = Arctic(conn)
    arctic_library = arctic[library]
    return arctic_library

# %% ../notebooks/07_arctic.ipynb 19
@dataclass
class ArcticLibraryInfo:
    ticker: str
    dates_ndarray: np.ndarray
    dates_series: pd.Series

    def __post_init__(self):
        self.dates_list: list[str] = list(self.dates_ndarray)
        self.start_date = min(self.dates_ndarray)
        self.end_date = max(self.dates_ndarray)

# %% ../notebooks/07_arctic.ipynb 20
CONTEXT_SETTINGS = dict(
    help_option_names=["-h", "--help"],
    token_normalize_func=lambda x: x.lower() if isinstance(x, str) else x,
    show_default=True,
    auto_envvar_prefix="ARCTIC"
)

# %% ../notebooks/07_arctic.ipynb 21
# REFACTORINOOOOO


def get_library_info(
    arctic_library: Library,  # arcticdb library
    tickers: list[str] | None = None,  # tickers to filter on
) -> list[ArcticLibraryInfo]:
    """Return information about ticker info in database."""

    arctic_symbols = arctic_library.list_symbols()
    if tickers:
        if not set(tickers).issubset(set(arctic_symbols)):
            raise ValueError(
                f"Some of the tickers specified were not in the databasee. The invalid tickers were {set(tickers) - set(arctic_symbols)}"
            )
    else:
        tickers = arctic_symbols

    arctic_library_infos: list[ArcticLibraryInfo] = []
    for ticker in tickers:
        q = QueryBuilder()
        # there is one auction each morning
        q = q[q.event == Event.CROSS_TRADE.value]
        df = arctic_library.read(symbol=ticker, query_builder=q).data

        dates_series: pd.Series = df.index.date
        dates_ndarray: np.ndarray = df.index.to_series().dt.strftime("%Y-%m-%d").values
        arctic_library_infos.append(
            ArcticLibraryInfo(
                ticker=ticker, dates_ndarray=dates_ndarray, dates_series=dates_series
            )
        )
    return arctic_library_infos


class Options:
    def __init__(self) -> None:
        self.db_path = click.option(
            "-d", "--db_path", default=cfg.db.db_path, help="Database path"
        )
        self.library = click.option(
            "-l", "--library", default=cfg.db.library, help="Library name"
        )
        self.ticker = click.option(
            "-t", "--ticker", required=True, help="ticker to print"
        )
        self.start_date = click.option(
            "-s", "--start_date", default=None, help="start date"
        )
        self.end_date = click.option("-e", "--end_date", default=None, help="end date")
        self.csv_path = click.option(
            "-c",
            "--csv_path",
            default=cfg.data_config.csv_files_path,
            help="csv files path",
        )
        self.etf = click.option(
            "--etf", default=None, help="restrict to subset specified by ETF members"
        )
        self.zip_path = click.option(
            "-z",
            "--zip_path",
            default="/nfs/lobster_data/lobster_raw/2016",
            help="zip files path",
        )
        self.tickers = click.option(
            "--tickers", default=None, multiple=True, type=str, help="tickers to dump"
        )
        self.max_workers = click.option(
            "-m", "--max_workers", default=20, help="max workers for parallelisation"
        )


O = Options()


class ConsoleNotify:
    def warn(self):
        click.secho("WARNING:", fg="red", bold=True, underline=True)

    def info(self):
        click.secho("INFO:", fg="yellow", bold=True, underline=True)

    def sucess(self):
        click.secho("SUCESS", fg="green", bold=True, underline=True)

C = ConsoleNotify()


def apply_options(options: list):
    def decorator(f):
        for option in reversed(options):
            f = option(f)
        return f

    return decorator


class ClickCtxObj(TypedDict):
    """Purely for type hinting. for instance `arctic_library` not always there."""

    library: str
    db_path: str
    arctic: Arctic
    arctic_library: NotRequired[Library]


class ClickCtx(Protocol):
    obj: ClickCtxObj


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-d", "--db_path", default=cfg.db.db_path, envvar="DB_PATH", help="Database path"
)
@click.option(
    "-l", "--library", default=cfg.db.library, envvar="LIBRARY", help="Library name"
)
@click.pass_context
def arctic(ctx, db_path, library):
    ctx.ensure_object(dict)
    arctic = Arctic(f"lmdb://{db_path}")
    ctx.obj.update(
        {
            "arctic": arctic,
            "library": library,
            "db_path": db_path,
        }
    )
    try:
        ctx.obj["arctic_library"] = arctic[library]
    except LibraryNotFound:
        pass


@arctic.command()
@click.pass_context
def echo(ctx: ClickCtx) -> None:
    """Echo back inputs"""
    click.echo(pformat(ctx.obj))


# not a good idea ! monkey patch click.echo to accept a color
# def new_echo(text, file=None, nl=True, err=False, color=None):
#     if color:
#         text = click.style(text, fg=color)
#     click.echo.original(text, file=file, nl=nl, err=err)

# def warn(msg):
#     return click.style(msg, fg="red", bold=True, blink=True)


@arctic.command()
def init():
    """Initialise autocomplete for arctic CLI"""
    # TODO: improve performance of CLI
    os.system("_ARCTIC_COMPLETE=bash_source arctic > ~/.arctic-complete.bash")

    with open(os.path.expanduser("~/.bashrc"), "a") as f:
        f.write(
            textwrap.dedent(
                """\
                # >>> arctic init_autocomplete >>>
                # Contents within this block were generated by arctic init_autocomplete
                . ~/.arctic-complete.bash
                # <<< arctic init_autocomplete <<<
                """
            )
        )

    click.echo(
        "Autocomplete initialized. Please restart your shell or run `source ~/.bashrc`."
    )
    click.echo(
        "Autocomplete initialized. Please restart your shell or run `source ~/.bashrc`."
    )


@arctic.command()
@click.pass_context
def create(ctx: ClickCtx) -> None:
    """Create a blank library"""
    arctic = ctx.obj["arctic"]
    library = ctx.obj["library"]
    arctic.create_library(library)
    click.echo(arctic[library])


@arctic.group()
@click.pass_context
def ls(ctx: ClickCtx):
    "List information about a library"
    # NOTE: Using word list clashed with python type hints!"""
    pass


@ls.command()
@click.pass_context
def libraries(ctx: ClickCtx):
    arctic = ctx.obj["arctic"]
    click.echo(arctic.list_libraries())


@ls.command()
@click.pass_context
def symbols(ctx: ClickCtx):
    arctic_library = ctx.obj["arctic_library"]
    click.echo(arctic_library.list_symbols())


@ls.command()
@click.pass_context
def versions(ctx: ClickCtx):
    arctic_library = ctx.obj["arctic_library"]

    click.echo(
        (
            pd.DataFrame(arctic_library.list_versions())
            .transpose()
            .drop(columns=[1, 2])
            .rename(columns={0: "created_on"})
            .assign(
                created_on=lambda df: df["created_on"].dt.strftime("%Y-%m-%d %H:%M:%S")
            )
            .rename_axis(["ticker", "version"])
            .sort_index(level=[0, 1], ascending=[True, False])
        )
    )


def parse_comma_separated(ctx, param, value: str):
    """Convert a comma (or space) separated option to a list of options"""
    if value is not None:
        delimiters = r"[ ,]"
        option_list = re.split(delimiters, value)
        option_list = list(filter(None, option_list))
        return option_list


@ls.command()
# @click.option("-t", "--tickers", callback=parse_comma_separated , help="Comma or space separated tickers")
@click.option(
    "-t", "--tickers", multiple=True, type=str, help="Provide ticker(s) to filter on"
)
@click.option(
    "-a",
    "--all",
    is_flag=True,
    default=False,
    help="print all dates not just start and end",
)
@click.pass_context
def dates(ctx: ClickCtx, tickers, all):
    arctic_library = ctx.obj["arctic_library"]

    arctic_library_infos = get_library_info(arctic_library, tickers=tickers)

    if all:
        click.echo(pformat({x.ticker: x.dates_list for x in arctic_library_infos}))
    else:
        click.echo(
            pformat(
                {x.ticker: (x.start_date, x.end_date) for x in arctic_library_infos}
            )
        )


@arctic.group()
@click.pass_context
def rm(ctx: ClickCtx):
    "Remove commands"
    # NOTE: Using word del clashed with python!
    pass


# TODO: make library an argument here rather than in arctic
@rm.command()
@click.pass_context
# @click.option(
#     "-l",
#     "--library",
#     required=True,
#     help="Confirm the library which you wish to delete",
# )
def library(ctx: ClickCtx):
    arctic = ctx.obj["arctic"]
    library = ctx.obj["library"]

    if not arctic.has_library(library):
        click.echo("No library found to delete.")
    else:
        arctic_library = arctic[library]
        C.info()
        click.echo(
            textwrap.dedent(
                f"""\
                Library information:
                {arctic_library}

                Tickers in this library:
                {arctic_library.list_symbols()}"""))
        C.warn()

        confirmation = click.prompt(f"Type {library} to confirm the permanent deletion of the library")        
        if confirmation == library:
            del ctx.obj["arctic_library"]
            del arctic_library
            arctic.delete_library(library)
            C.sucess()
        else:
            raise click.Abort()


@arctic.command()
@click.argument("etf")
def etf(etf):
    "Output constituents of ETF including the ETF itself"
    click.echo("\n".join([etf] + etf_to_equities[etf]))

@arctic.command()
@click.argument("query_template")
def query(query_template: str):
    "Write a custom query"
    for line in sys.stdin:
        obj = json.loads(line.strip())
        query = Template(query_template).substitute(obj)
        click.echo(query)

# @arctic.command()
# @click.argument("query_template")
# def prep(query_template: str):
#     "Write a custom query"
#     for line in sys.stdin:
#         obj = json.loads(line.strip())
#         query = Template(query_template).substitute(obj)
#         click.echo(query)

@arctic.command()
@click.option(
    "-f",
    "--files_path",
    default=cfg.data_config.csv_files_path,
    help="files path",
)
def finfo(files_path):
    "Output "
    l = infer_ticker_dict(files_path)
    l = [asdict(x) for x in l]
    # l = [x.append()]
    l = [json.dumps(x) for x in l]
    # l = [json.dumps(asdict(x)) for x in l]
    click.echo("\n".join(l))

@arctic.command()
@click.pass_context
@click.option(
    "-s", "--start_date", default=None, help="start date"
)
@click.option("-e", "--end_date", default=None, help="end date")
@click.option(
    "-c",
    "--csv_path",
    default=cfg.data_config.csv_files_path,
    help="csv files path",
)
@click.option(
    "-z",
    "--zip_path",
    default="/nfs/lobster_data/lobster_raw/2016",
    help="zip files path",
)
def add(ctx, start_date, end_date, csv_path, zip_path):
    "Add extra fields to JSON objects read from stdin"
    for line in sys.stdin:
        obj = json.loads(line.strip())
        
        obj["library"] = ctx.obj["library"]
        obj["db_path"] = ctx.obj["db_path"]
        obj['csv_path'] = csv_path
        obj['zip_path'] = zip_path
        if start_date: obj['start_date'] = start_date
        if end_date: obj['end_date'] = end_date

        # Output the updated JSON object
        click.echo(json.dumps(obj))


@arctic.command()
@click.pass_context
@click.option(
    "-c",
    "--csv_path",
    default=cfg.data_config.csv_files_path,
    help="csv files path",
)
@click.option(
    "--ticker",
    required=True,
)
@click.option(
    "--start_date",
)
@click.option(
    "--end_date",
)
def write(
    ctx,
    csv_path,
    ticker,
    start_date,
    end_date,
):
    """Single thread write ticker to database."""
    # ok maybe slightly confusing to read ticker, start_date, end_date from stdin
    # but other things from ctx.obj
    try:
        arctic_library = ctx.obj["arctic_library"]
    except KeyError:
        raise LibraryNotFound
    # TODO: as of now only valid if both start and end are provided
    if bool(start_date) ^ bool(end_date):
        raise NotImplementedError 
    date_range = (start_date, end_date) if start_date else None

    data = Data(
        directory_path=csv_path,
        ticker=ticker,
        date_range=date_range,
        aggregate_duplicates=False,
    )
    lobster = Lobster(data=data)
    df = pd.concat([lobster.messages, lobster.book], axis=1)
    C.info()
    print(f"head of ticker {ticker}")
    print(df.head())

    arctic_library.write(symbol=ticker, data=df)

    C.sucess()

# @arctic.command()
# @apply_options([O.db_path, O.library, O.csv_path, O.ticker, O.start_date, O.end_date])
# def write(**kwargs):
#     _write(**kwargs)


# %% ../notebooks/07_arctic.ipynb 31
# | code-fold: true
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-d", "--db_path", default=cfg.db.db_path, help="database path")
@click.option("-l", "--library", default=cfg.db.library, help="library name")
def arctic_list_symbols(db_path, library) -> None:
    """List symbols in the arcticdb library."""
    arctic_library = get_arctic_library(db_path=db_path, library=library)
    print(f"Symbols in library {library}")
    print(arctic_library.list_symbols())

# %% ../notebooks/07_arctic.ipynb 32
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

# %% ../notebooks/07_arctic.ipynb 33
# | code-fold: true
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-d", "--db_path", default=cfg.db.db_path, help="database path")
def arctic_list_libraries(db_path) -> None:
    """List arcticdb libraries"""

    conn = f"lmdb://{db_path}"
    arctic = Arctic(conn)
    print(arctic.list_libraries())

# %% ../notebooks/07_arctic.ipynb 34
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

# %% ../notebooks/07_arctic.ipynb 35
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

# %% ../notebooks/07_arctic.ipynb 37
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

# %% ../notebooks/07_arctic.ipynb 38
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

# %% ../notebooks/07_arctic.ipynb 39
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

# %% ../notebooks/07_arctic.ipynb 40
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
