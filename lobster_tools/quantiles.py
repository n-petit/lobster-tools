from pathlib import Path

import pandas as pd
from absl import app, flags

import lobster_tools.config  # noqa: F401

_ALL_QUANTILES = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995]

FLAGS = flags.FLAGS


def main(_):
    # set output directory
    output_dir = Path(FLAGS.output_dir)
    etf_dir = output_dir / "etf" / FLAGS.etf

    for epsilon in FLAGS.epsilons:
        eps_dir = etf_dir / "epsilon" / epsilon

        # load data
        features = pd.read_pickle(eps_dir / "features.pkl")

        # compute quantiles
        quantiles = features.groupby(features.index.date).apply(
            lambda x: x[x != 0].quantile(_ALL_QUANTILES)
        )
        quantiles.index = quantiles.index.set_names(["date", "quantile"])

        # save to file
        quantiles.to_pickle(eps_dir / "quantiles.pkl")


if __name__ == "__main__":
    app.run(main)
