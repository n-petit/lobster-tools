"""Compute Conditional Order Imbalances (COIs) for both decomposed order flow (i.e 
tagged) and for untagged.
"""

import itertools as it
from pathlib import Path

import pandas as pd
from absl import app, flags, logging

import lobster_tools.config  # noqa: F401
from lobster_tools.utils import get_dir

FLAGS = flags.FLAGS


# TODO: if re-running this should change to COI for **aggressive** orders (see below).
def conditional_order_imbalance(group: pd.DataFrame, resample_freq: str):
    "Compute COI (Conditional Order Imbalance) for a sequence of trades."
    return (
        group[["size", "direction"]]
        .eval("signed_size = size * direction")  # TODO: here would be an extra * -1
        .drop(columns=["direction"])
        .resample(resample_freq, label="right", closed="left")
        .sum()
        .eval("ofi = signed_size / size")
        .ofi.fillna(0)
    )


def main(_):
    # load data
    logging.info("loading data")

    dir_ = get_dir(etf=FLAGS.etf)
    etf_executions = pd.read_pickle(dir_ / "etf_executions.pkl")

    # COIs for tagged flow
    for epsilon, resample_freq in it.product(FLAGS.epsilons, FLAGS.resample_freqs):
        logging.info(
            f"computing COIs for epsilon={epsilon} and resample_freq={resample_freq}"
        )

        dir_ = get_dir(
            etf=FLAGS.etf,
            epsilon=epsilon,
            quantiles=FLAGS.quantiles,
            lookback=FLAGS.lookback,
        )
        tags = pd.read_pickle(dir_ / "tags.pkl")

        etf_executions = etf_executions.drop(columns="notional")
        full = etf_executions.join(tags, how="inner")

        # COI calculation for each feature for each tag
        cois_dict = {}
        for col in tags.columns:
            logging.info(f"computing coi for {col}")
            cois = full.groupby([full.index.date, full[col]], observed=False).apply(
                conditional_order_imbalance, resample_freq=resample_freq
            )
            cois.index = cois.index.droplevel(0)
            cois = cois.unstack(level=0)
            cois = cois.fillna(0)
            cois.columns = cois.columns.tolist()
            cois_dict[col] = cois

        # write to disk
        dir_ = get_dir(
            etf=FLAGS.etf,
            epsilon=epsilon,
            resample_freq=resample_freq,
            quantiles=FLAGS.quantiles,
            lookback=FLAGS.lookback,
        )
        dir_.mkdir(parents=True, exist_ok=True)

        pd.to_pickle(cois_dict, dir_ / "cois.pkl")

    # COI for untagged flow
    logging.info("computing COI for untagged flow")
    for resample_freq in FLAGS.resample_freqs:
        logging.info(f"computing COI for resample_freq={resample_freq}")

        coi_no_tagging = etf_executions.groupby(
            etf_executions.index.date, observed=False
        ).apply(conditional_order_imbalance, resample_freq=resample_freq)
        coi_no_tagging.index = coi_no_tagging.index.droplevel(0)
        # no unstacking for single column
        coi_no_tagging = coi_no_tagging.fillna(0)

        # write to disk
        dir_ = get_dir(etf=FLAGS.etf, resample_freq=resample_freq)
        dir_.mkdir(parents=True, exist_ok=True)

        pd.to_pickle(coi_no_tagging, dir_ / "coi_no_tagging.pkl")


if __name__ == "__main__":
    app.run(main)
