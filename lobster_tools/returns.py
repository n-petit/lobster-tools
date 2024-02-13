from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging

from lobster_tools import flow

flags.DEFINE_string(
    "resample_freq",
    "5T",
    "Resample frequency.",
)

flags.DEFINE_list(
    "markouts",
    ["30S", "1T", "2T", "5T", "20T", "60T", "120T"],
    "Markouts.",
)

# _MARKOUTS = ['1/2', '1', '2', '4']
# flags.DEFINE_multi_float(
#     "markouts",
#     [0.5, 1.0, 2.0, 4.0],
#     "Markouts as a function of the resample freq.",
# )

FLAGS = flags.FLAGS


def resampled_log_returns(group):
    markouts = FLAGS.markouts
    resample_freq = FLAGS.resample_freq

    s = group.price.resample(
        resample_freq, label="right", closed="right", offset="0S"
    ).last()
    log_s = np.log(s)

    cols = ["contemp"] + [f"fRet_{markout}" for markout in markouts]
    df = pd.DataFrame(index=s.index, columns=cols)

    df["contemp"] = log_s - log_s.shift(1)

    for markout in markouts:
        offset = group.price.resample(
            resample_freq, label="right", closed="right", offset=markout
        ).last()
        offset.index = offset.index - pd.Timedelta(markout)
        df[f"fRet_{markout}"] = np.log(offset) - log_s

    return df


@dataclass
class ResampleFreq:
    _str: str
    timedelta: pd.Timedelta = field(init=False)

    def __post_init__(self):
        self.timedelta = pd.Timedelta(self._str)


_OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def main(_):
    # resample_freq = ResampleFreq(FLAGS.resample_freq)
    # markouts = [float(markout) * resample_freq.timedelta for markout in FLAGS.markouts]
    logging.info(f"Resample freq: {FLAGS.resample_freq}")
    logging.info(f"ETF: {FLAGS.etf}")

    etf_dir = _OUTPUT_DIR / "latest" / FLAGS.etf

    # TODO: load from arctic not only trades
    etf_executions = pd.read_pickle(etf_dir / "etf_executions.pkl")
    logging.info(etf_executions.head())

    returns = etf_executions.groupby(etf_executions.index.date).apply(
        resampled_log_returns
    )
    returns.index = returns.index.droplevel(0)
    returns = returns.dropna()
    logging.info(returns.head())

    # save to disk
    resample_dir = etf_dir / "resample"
    resample_dir.mkdir(exist_ok=True)
    returns.to_pickle(resample_dir / f"{FLAGS.resample_freq}.pkl")

    # TODO: hedged returns

def echo(_):
    print(FLAGS.resample_freq)
    print(FLAGS.etf)

if __name__ == "__main__":
    app.run(main)
