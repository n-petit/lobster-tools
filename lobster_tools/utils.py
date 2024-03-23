from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from absl import flags, app
from beartype import beartype

from lobster_tools import config  # noqa: F401

FLAGS = flags.FLAGS


@beartype
def get_dir(
    etf: str,
    epsilon: str = None,
    resample_freq: str = None,
    quantiles: list[float] = None,
    lookback: int = None,
) -> Path:
    "Return the output directory for the given hyperparameters."

    params = {
        "etf": etf,
        "epsilon": epsilon,
        "resample_freq": resample_freq,
        "quantiles": quantiles,
        "lookback": lookback,
    }
    params = {k: v for k, v in params.items() if v is not None}

    if "quantiles" in params:
        params["quantiles"] = "_".join(map(str, params["quantiles"]))
    if "lookback" in params:
        params["lookback"] = str(params["lookback"])

    sort_order = ("etf", "epsilon", "quantiles", "lookback", "resample_freq")
    sorted_params = sorted(params.items(), key=lambda x: sort_order.index(x[0]))
    params = dict(sorted_params)  # python 3.7+ maintains insertion order

    path = Path(FLAGS.output_dir)
    for k, v in params.items():
        path /= k
        path /= v

    return path
