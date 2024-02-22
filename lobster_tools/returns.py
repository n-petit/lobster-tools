import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging

from lobster_tools import config  # noqa: F401


# markouts as fractions of the resample_freq
MARKOUTS = [Decimal(x) for x in ["0.125", "0.25", "0.5", "1.0", "2.0", "4.0"]]

FLAGS = flags.FLAGS


def resampled_log_returns(group, resample_freq: str, markouts: list[Decimal]):
    """Markouts computed from base and markout series. Base timeseries using last value
    in ( ] and stamped right. Marked out price is first value of shifted ( ] stamped
    left."""
    base_offset = pd.Timedelta("0S")
    base_resampled = group.price.resample(
        resample_freq, label="right", closed="right", offset=base_offset
    ).last()
    log_base_resampled = np.log(base_resampled)

    cols = ["contemp"] + [f"fRet_{markout}" for markout in markouts] + ["fRet_close"]
    df = pd.DataFrame(index=base_resampled.index, columns=cols)

    df["contemp"] = log_base_resampled - log_base_resampled.shift(1)

    closing_price = group.price.resample("D").last().iat[-1]
    df["fRet_close"] = np.log(closing_price) - log_base_resampled
    df.loc[df.index[-1], "fRet_close"] = np.NaN  # last entry is always 0

    for markout in markouts:
        offset = float(markout) * pd.Timedelta(resample_freq)
        offset_resampled = group.price.resample(
            resample_freq, label="left", closed="right", offset=offset
        ).first()
        offset_resampled.index = offset_resampled.index - offset
        df[f"fRet_{markout}"] = np.log(offset_resampled) - log_base_resampled

    return df


def main(_):
    # output directories
    output_dir = Path(FLAGS.output_dir)
    etf_dir = output_dir / "etf" / FLAGS.etf

    logging.get_absl_handler().use_absl_log_file(log_dir=etf_dir)
    sys.stderr = logging.get_absl_handler().python_handler.stream

    logging.info(f"FLAGS: {FLAGS}")

    # TODO: load from arctic and use mid
    etf_executions = pd.read_pickle(etf_dir / "etf_executions.pkl")

    for resample_freq in FLAGS.resample_freqs:
        resample_dir = etf_dir / "resample_freq" / resample_freq
        resample_dir.mkdir(parents=True, exist_ok=True)

        returns = etf_executions.groupby(etf_executions.index.date).apply(
            resampled_log_returns,
            resample_freq=resample_freq,
            markouts=MARKOUTS,
        )
        returns.index = returns.index.droplevel(0)

        # save to disk
        returns.to_pickle(resample_dir / "returns.pkl")

        # TODO: hedged returns


if __name__ == "__main__":
    app.run(main)
