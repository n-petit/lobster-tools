# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/06_data_downloading.ipynb.

# %% auto 0
__all__ = ['CONTEXT_SETTINGS', 'get_sample_data']

# %% ../notebooks/06_data_downloading.ipynb 4
import io
import os
import zipfile
from typing import Literal

import click
import requests

# %% ../notebooks/06_data_downloading.ipynb 5
# | code-fold: true
CONTEXT_SETTINGS = dict(
    help_option_names=["-h", "--help"],
    token_normalize_func=lambda x: x.lower() if isinstance(x, str) else x,
)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-t", "--ticker", default="AMZN", help="ticker")
@click.option("-l", "--levels", default=5, help="number of levels")
def get_sample_data(
    ticker: Literal["AMZN", "AAPL", "GOOG", "INTC", "MSFT"],
    levels: Literal[1, 5, 10],
    output_dir=None,
):
    """Download and extract sample data from LOBSTER website."""
    SAMPLE_DATA_DATE = "2012-06-21"
    url = f"https://lobsterdata.com/info/sample/LOBSTER_SampleFile_{ticker}_{SAMPLE_DATA_DATE}_{levels}.zip"
    print(f"Downloading data from {url}")
    
    default_directory_name = f"data/{ticker}_{SAMPLE_DATA_DATE}_{SAMPLE_DATA_DATE}_{levels}"
    output_dir = os.path.join(os.getcwd(), default_directory_name)

    os.makedirs(output_dir, exist_ok=True)

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to download data. HTTP Status Code: {response.status_code}")
        return

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(output_dir)
