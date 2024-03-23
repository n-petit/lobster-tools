from decimal import Decimal

import numpy as np
import pandas as pd
from absl import app, flags

from lobster_tools import config  # noqa: F401
from lobster_tools.utils import get_dir

FLAGS = flags.FLAGS

# markouts as fractions of the resample frequency `resample_freq'
MARKOUTS = [Decimal(x) for x in ["0.125", "0.25", "0.5", "1.0", "2.0", "4.0"]]


def get_log_returns(group, resample_freq: str, markouts: list[Decimal]):
    """Compute resampled log returns. Markouts computed from base and markout series. 
    Base timeseries using last value in ( ] and stamped right. Marked out price is first
    value of shifted ( ] stamped left."""
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
    # TODO: load from arctic and use mid to mid
    dir_ = get_dir(etf=FLAGS.etf)
    etf_executions = pd.read_pickle(dir_ / "etf_executions.pkl")

    for resample_freq in FLAGS.resample_freqs:
        # intraday log returns
        returns = (
            etf_executions.groupby(etf_executions.index.date)
            .apply(
                get_log_returns,
                resample_freq=resample_freq,
                markouts=MARKOUTS,
            )
            .droplevel(0)
        )

        # save to disk
        dir_ = get_dir(
            etf=FLAGS.etf,
            resample_freq=resample_freq,
        )
        dir_.mkdir(exist_ok=True)
        returns.to_pickle(dir_ / "returns.pkl")

        # TODO: hedged returns


if __name__ == "__main__":
    app.run(main)
