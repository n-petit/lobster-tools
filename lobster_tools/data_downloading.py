__all__ = ['cfg', 'CONTEXT_SETTINGS', 'get_sample_data']

import io
import os
import zipfile
from typing import Literal

import click
import requests

from lobster_tools.config import get_config

cfg = get_config()

CONTEXT_SETTINGS = dict(
    help_option_names=["-h", "--help"],
    token_normalize_func=lambda x: x.lower() if isinstance(x, str) else x,
    show_default=True,
)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-t", "--ticker", default=cfg.sample_data.ticker, help="ticker")
@click.option("-l", "--levels", default=cfg.sample_data.levels, help="number of levels")
def get_sample_data(
    ticker: Literal["AMZN", "AAPL", "GOOG", "INTC", "MSFT"],
    levels: Literal[1, 5, 10],
):
    """Download and extract sample data from LOBSTER website."""
    SAMPLE_DATA_DATE = "2012-06-21"
    url = f"https://lobsterdata.com/info/sample/LOBSTER_SampleFile_{ticker}_{SAMPLE_DATA_DATE}_{levels}.zip"
    print(f"Downloading data from {url}")

    default_directory_name = (
        f"data/{ticker}_{SAMPLE_DATA_DATE}_{SAMPLE_DATA_DATE}_{levels}"
    )
    output_dir = os.path.join(os.getcwd(), default_directory_name)

    os.makedirs(output_dir, exist_ok=True)

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to download data. HTTP Status Code: {response.status_code}")
        return

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(output_dir)