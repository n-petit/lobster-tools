__all__ = ['cfg', 'CONTEXT_SETTINGS', 'O', 'C', 'ArcticLibraryInfo', 'get_library_info', 'Options', 'ConsoleNotify',
           'ClickCtxObj', 'ClickCtx', 'etf', 'pfmt', 'arctic', 'echo', 'create', 'ls', 'libraries', 'symbols',
           'versions', 'dates', 'rm', 'library', 'query', 'filter', 'finfo', 'attach', 'diff', 'single_write',
           'process_ticker', 'dump']

import json
import socket
import subprocess
import sys
import textwrap
import typing as t
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pprint import pformat
from string import Template
from typing import NotRequired, Protocol

import click
import numpy as np
import pandas as pd
from arcticdb import Arctic, QueryBuilder
from arcticdb.exceptions import LibraryNotFound
from arcticdb.version_store.library import Library

from lobster_tools.config import (
    Overrides,
    etf_to_equities,
    get_config,
)
from lobster_tools.preprocessing import (
    Data,
    Event,
    Lobster,
    MPLobster,
    infer_ticker_date_ranges,
)

cfg = get_config(overrides=Overrides.full_server)

@dataclass
class ArcticLibraryInfo:
    ticker: str
    dates_ndarray: np.ndarray
    dates_series: pd.Series

    def __post_init__(self):
        self.dates_list: list[str] = list(self.dates_ndarray)
        self.start_date = min(self.dates_ndarray)
        self.end_date = max(self.dates_ndarray)

CONTEXT_SETTINGS = dict(
    help_option_names=["-h", "--help"],
    token_normalize_func=lambda x: x.lower() if isinstance(x, str) else x,
    show_default=True,
    auto_envvar_prefix="ARCTIC",
)

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
            default="/nfs/lobster_data/lobster_raw/2021",
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


class ClickCtxObj(t.TypedDict):
    "Purely for type hinting. for instance `arctic_library` not always there."

    library: str
    db_path: str
    arctic: Arctic
    arctic_library: NotRequired[Library]


class ClickCtx(Protocol):
    obj: ClickCtxObj


@click.command()
@click.argument("etf")
@click.option("-s", "--sep", default="\n", help="separator")
def etf(etf, sep):
    "Output constituents of ETF including the ETF itself"
    click.echo(sep.join([etf] + etf_to_equities[etf]))


@click.command()
def pfmt():
    "Simple jq like utility to pretty format json objects."
    for line in sys.stdin:
        obj = json.loads(line.strip())
        click.echo(pformat(obj))


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-d", "--db_path", default=cfg.db.db_path, envvar="DB_PATH", help="Database path"
)
@click.option(
    "-l", "--library", default=cfg.db.library, envvar="LIBRARY", help="Library name"
)
@click.option("--s3", is_flag=True, default=True, help="Use s3 bucket")
@click.pass_context
def arctic(ctx, db_path, library, s3):
    ctx.ensure_object(dict)
    if s3:
        arctic = Arctic(
            "s3://163.1.179.45:9100:lobster?access=minioadmin&secret=minioadmin"
        )
    else:
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
    "Debugging tool that echoes back the arctic object."
    click.echo(pformat(ctx.obj))


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


@ls.command()
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
    "Remove."
    pass


@rm.command()
@click.pass_context
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
                {arctic_library.list_symbols()}"""
            )
        )
        C.warn()

        confirmation = click.prompt(
            f"Type {library} to confirm the permanent deletion of the library"
        )
        if confirmation == library:
            del ctx.obj["arctic_library"]
            del arctic_library
            arctic.delete_library(library)
            C.sucess()
        else:
            raise click.Abort()


@arctic.command()
@click.argument("query_template")
def query(query_template: str):
    "Write a custom query using a string template. Reads json objects from stdin and writes queries to stdout."
    for line in sys.stdin:
        obj = json.loads(line.strip())
        query = Template(query_template).substitute(obj)
        click.echo(query)


@arctic.command()
@click.argument("tickers", nargs=-1)
def filter(tickers: tuple):
    "Filter by ticker. Reads json objects from stdin and writes filtered objects to stdout."
    for line in sys.stdin:
        obj = json.loads(line.strip())
        if obj["ticker"] in tickers:
            click.echo(json.dumps(obj))


@arctic.command()
@click.option(
    "-f",
    "--files_path",
    default=cfg.data_config.csv_files_path,
    help="files path",
)
def finfo(files_path):
    "Output json objects with folder information."
    l = infer_ticker_date_ranges(files_path)
    l = [asdict(x) for x in l]
    l = [json.dumps(x) for x in l]
    click.echo("\n".join(l))


@arctic.command()
@click.pass_context
@click.option("-s", "--start_date", envvar="ARCTIC_START_DATE", help="start date")
@click.option("-e", "--end_date", envvar="END_DATE", help="end date")
@click.option(
    "-c",
    "--csv_path",
    default=cfg.data_config.csv_files_path,
    envvar="CSV_PATH",
    help="csv files path",
)
@click.option(
    "-z",
    "--zip_path",
    default="/nfs/lobster_data/lobster_raw/2021",
    envvar="ZIP_PATH",
    help="zip files path",
)
def attach(ctx, start_date, end_date, csv_path, zip_path):
    "Attach extra matadata to JSON objects read from stdin."
    for line in sys.stdin:
        obj = json.loads(line.strip())

        obj["library"] = ctx.obj["library"]
        obj["db_path"] = ctx.obj["db_path"]
        obj["csv_path"] = csv_path
        obj["zip_path"] = zip_path

        if start_date:
            obj["start_date"] = start_date
        if end_date:
            obj["end_date"] = end_date

        click.echo(json.dumps(obj))


@arctic.command()
@click.pass_context
@click.option(
    "-z",
    "--zip_path",
    default="/nfs/lobster_data/lobster_raw/2021",
    help="zip files path",
)
def diff(
    ctx: ClickCtx,
    zip_path: str,
):
    "Tickers still to be written to database."
    arctic_library = ctx.obj["arctic_library"]

    csv_info = infer_ticker_date_ranges(zip_path)
    csv_tickers = [x.ticker for x in csv_info]

    arctic_tickers = arctic_library.list_symbols()

    tickers_difference = set(csv_tickers).difference(arctic_tickers)
    click.echo(tickers_difference)


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
@click.option("--date_range", nargs=2, type=str)
@click.option(
    "--update", is_flag=True, default=False, help="use update instead of write"
)
@click.option(
    "--mp",
    is_flag=True,
    default=True,
    help="Use multiprocessing to load data.",
)
def single_write(
    ctx: ClickCtx,
    csv_path: str,
    ticker: str,
    date_range: tuple,
    update: bool,
    mp: bool,
):
    """Write single ticker to database."""
    try:
        arctic_library = ctx.obj["arctic_library"]
    except KeyError:
        raise LibraryNotFound

    data = Data(
        directory_path=csv_path,
        ticker=ticker,
        date_range=date_range,
        load="both",
        aggregate_duplicates=False,
    )
    click.echo(data)

    if mp:
        lobster = MPLobster(data=data)
    else:
        lobster = Lobster(data=data)

    df = pd.concat([lobster.messages, lobster.book], axis=1)
    C.info()
    print(f"head of ticker {ticker}")
    print(df.head())

    if update:
        # for batched writes for large tickers like SPY
        arctic_library.update(symbol=ticker, data=df)
    else:
        arctic_library.write(symbol=ticker, data=df)
    C.sucess()

def process_ticker(ticker, ticker_till_end, full):
    """Chain of mkdir, unzip, write to arctic db, remove tmp folder."""

    tmp_dir = f"/nfs/home/nicolasp/home/data/tmp/{ticker_till_end}"
    raw_data = f"{full}"

    subprocess.run(["mkdir", tmp_dir])

    subprocess.run(["7z", "x", raw_data, f"-o{tmp_dir}"])

    subprocess.run(
        ["arctic", "--library=demo", "single-write", f"--ticker={ticker}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    subprocess.run(["rm", "-rf", tmp_dir])

def _echo(ticker, ticker_till_end, full):
    "Echo back function inputs"
    print(ticker, ticker_till_end, full)

@arctic.command()
@click.option(
    "-t", "--tickers", multiple=True, type=str, help="Tickers to add to database."
)
@click.option(
    "-s", "--server_numbers", multiple=True, type=str, help="RAPID servers to run on."
)
@click.option(
    "--max_workers_per_server", type=int, help="Max number of workers per server."
)
def dump(
    server_numbers,
    tickers,
    max_workers_per_server=1,
):
    host_name = socket.gethostname()
    print(f"starting on hostname {host_name}")
    finfo = infer_ticker_date_ranges("/nfs/lobster_data/lobster_raw/2021")

    # tickers = [
    #     "XLB",
    #     "MSFT",
    #     "XLK",
    #     "XLY",
    #     "AMD",
    # ]
    finfo = [x for x in finfo if x.ticker in tickers]

    # servers to split jobs to
    # server_numbers: list[str] = ["02", "18", "19", "20", "21"]
    servers = ["omi-rapid-" + x for x in server_numbers]

    job_chunks = np.array_split(finfo, len(servers))
    server_to_jobs = {
        server: job_chunk.tolist() for server, job_chunk in zip(servers, job_chunks)
    }
    jobs = server_to_jobs[host_name]
    tickers, tickers_till_end, full = zip(
        *[(f.ticker, f.ticker_till_end, f.full) for f in jobs]
    )

    with ProcessPoolExecutor(max_workers=max_workers_per_server) as executor:
        executor.map(process_ticker, tickers, tickers_till_end, full)
