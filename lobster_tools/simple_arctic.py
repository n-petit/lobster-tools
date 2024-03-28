from pathlib import Path
import subprocess

from absl import app, flags, logging
from arcticdb import Arctic
from arcticdb.version_store.library import Library
import pandas as pd
import ray

from lobster_tools import config  # noqa: F401
from lobster_tools.refactored_preprocessing import (
    MetadataFrom7zFile,
    infer_metadata_from_7z_files,
    infer_metadata_from_csv_files,
    load_messages,
    load_orderbooks,
)  # noqa: F401

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "zip_dir",
    default="/nfs/data/lobster_data/lobster_raw/2021",
    help="Path to the 7z zip files.",
)
flags.DEFINE_string(
    "tmp_dir",
    default="/nfs/home/nicolasp/home/data/tmp",
    help="Path to directory where files are unzipped.",
)


# old
@ray.remote
def unzip_and_write_chain(
    arctic_library: Library, file: MetadataFrom7zFile, tmp_dir: str
):
    ticker_dir = Path(tmp_dir) / file.ticker
    ticker_dir.mkdir(exist_ok=True)  # for tickers with several zip files per year

    subprocess.run(["7z", "x", file.zip_path, f"-o{ticker_dir}"])

    csv_files = infer_metadata_from_csv_files(ticker_dir)

    messages = load_messages(csv_files.messages)
    orderbooks = load_orderbooks(csv_files.orderbooks)

    assert len(messages) == len(orderbooks)
    df = pd.concat([messages, orderbooks], axis=1)

    arctic_library.write(symbol=file.ticker, data=df)


def main(_):
    arctic = Arctic(FLAGS.s3_uri)

    # for now just demo library
    # arctic_library = arctic[FLAGS.library]
    arctic_library = arctic["demo"]

    print(arctic_library.list_symbols())

    tmp_dir = Path(FLAGS.tmp_dir)
    tmp_dir.mkdir(exist_ok=True)

    print(tmp_dir)

    zip_files = infer_metadata_from_7z_files(FLAGS.zip_dir)

    zip_files = zip_files[:3]
    print(zip_files)


    @ray.remote
    def unzip_and_write_to_db(
        zip_file_metadata: MetadataFrom7zFile,
    ):
        logging.info('launch subjob')
        ticker_dir = tmp_dir / zip_file_metadata.ticker
        logging.info(f'ticker_dir: {ticker_dir}')
        # TODO: think
        ticker_dir.mkdir(exist_ok=True)  # for tickers with several zip files per year

        # subprocess.run(["7z", "x", zip_file_metadata.zip_path, f"-o{ticker_dir}"])

        csv_files = infer_metadata_from_csv_files(ticker_dir)
        logging.info(csv_files.orderbooks)
        logging.info(csv_files.messages)

        messages = load_messages(csv_files.messages)
        orderbooks = load_orderbooks(csv_files.orderbooks)

        assert len(messages) == len(orderbooks)
        df = pd.concat([messages, orderbooks], axis=1)
        logging.info(f'df head: {df.head()}')

        arctic_library.write(symbol=zip_file_metadata.ticker, data=df)
        logging.info(f'write complete for ticker: {zip_file_metadata.ticker}')

    # ray.init(dashboard_host='0.0.0.0', dashboard_port=1234)
    context = ray.init()
    print('hello wordl')
    logging.info(context.dashboard_url)

    ray.get(
        [
            unzip_and_write_to_db.remote(
                zip_file,
            )
            for zip_file in zip_files
        ]
    )
    ray.shutdown()

def list_ticker_in_demo_lib(_):
    arctic = Arctic(FLAGS.s3_uri)
    arctic_library = arctic["demo"]
    print(arctic_library.list_symbols())


def simple_ray_computations(_):
    @ray.remote
    def some_work(x):
        return [i**3 for i in range(x)]

    ray.init()
    res = ray.get(
        [
            some_work.remote(
                x,
            )
            for x in range(100)
        ]
    )
    logging.info(res)
    ray.shutdown()

if __name__ == "__main__":
    app.run(main)
    # app.run(simple_ray_computations)
    # app.run(list_ticker_in_demo_lib)
