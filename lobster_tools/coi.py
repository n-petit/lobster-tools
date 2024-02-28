"Compute Conditional Order Imbalances (COIs). Depends on tags.pkl"
import datetime as dt
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from lobster_tools.config import FEATURE_NAMES_WITH_MARGINALS

from lobster_tools.flow import evaluate_features

import seaborn as sns
import matplotlib.pyplot as plt

from lobster_tools import returns, tagging  # noqa: F401
import lobster_tools.config  # noqa: F401

FLAGS = flags.FLAGS


def coi(group: pd.DataFrame, resample_freq: str):
    "Compute COI (Conditional Order Imbalance) for a sequence of trades."
    return (
        group[["size", "direction"]]
        .eval("signed_size = size * direction")
        .drop(columns=["direction"])
        .resample(resample_freq, label="right", closed="left")
        .sum()
        .eval("ofi = signed_size / size")
        .ofi.fillna(0)
    )


def restrict_to_common_index(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict two dataframes to their common index."""
    common_index = df1.index.intersection(df2.index)
    return df1.loc[common_index], df2.loc[common_index]


def main(_):
    # load data
    logging.info("loading data")

    output_dir = Path(FLAGS.output_dir)
    etf_dir = output_dir / "etf" / FLAGS.etf

    # COIs for tagged flow
    for epsilon, resample_freq in itertools.product(FLAGS.epsilons, FLAGS.resample_freqs):
        logging.info(f"computing COIs for epsilon={epsilon} and resample_freq={resample_freq}")

        #######################################
        eps_dir = etf_dir / "epsilon" / epsilon
        quantile_dir = eps_dir / "quantiles" / "_".join(map(str, FLAGS.quantiles))
        lookback_dir = quantile_dir / "lookback" / str(FLAGS.lookback)

        etf_executions = pd.read_pickle(etf_dir / "etf_executions.pkl")
        tags = pd.read_pickle(lookback_dir / "tags.pkl")
        TAG_COLUMNS = tags.columns

        etf_executions = etf_executions.drop(columns="notional")
        full = etf_executions.join(tags, how="inner")

        # coi calculation
        cois_dict = {}
        for col in TAG_COLUMNS:
            logging.info(f"computing coi for {col}")
            cois = full.groupby([full.index.date, full[col]], observed=False).apply(
                coi, resample_freq=resample_freq
            )
            cois.index = cois.index.droplevel(0)
            cois = cois.unstack(level=0)
            cois = cois.fillna(0)
            cois.columns = cois.columns.tolist()
            cois_dict[col] = cois
        
        # write to disk
        # TODO: improve structure of writing files to disk
        resample2_dir = lookback_dir / "resample_freq" / resample_freq
        resample2_dir.mkdir(parents=True, exist_ok=True)

        pd.to_pickle(cois_dict, resample2_dir / "cois.pkl")

    return
    ##############################################################
    # moved to regressions.py
    ##############################################################

    # load returns data
    resample_dir = etf_dir / "resample_freq" / resample_freq
    returns = pd.read_pickle(resample_dir / "returns.pkl")

    # univariate regressions
    for feature, df in cois_dict.items():
        results = []
        for tag, ser in df.items():
            # TODO: only drop na if necessary
            ser = ser.dropna()
            returns = returns.dropna()
            X, Y = restrict_to_common_index(ser, returns)
            X = X.values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, Y)
            Y_hat: np.ndarray = model.predict(X)

            for i, markout in enumerate(Y.columns):
                y = Y[markout]
                y_pred = Y_hat[:, i]
                score = r2_score(y, y_pred)
                results.append({'feature': feature, 'tag': tag, 'markout': markout, 'score': score})
        results = pd.DataFrame(results)
        sns.catplot(data=results, x='tag', y='score', hue='markout', kind='bar')

        feature_dir = lookback_dir / "feature" / feature
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(feature_dir / f"univariate_regressions.png")
    print('hi')
    ###########################################
    ###########################################
    return
    ###########################################
    ###########################################

    # multivariate regression
    logging.info("running multivariate regressions")
    results_dict = {}
    for feature_name, cois in cois_dict.items():
        cois = cois.dropna()
        returns = returns.dropna()

        X, Y = restrict_to_common_index(cois, returns)
        model = LinearRegression()
        model.fit(X, Y)
        Y_hat: np.ndarray = model.predict(X)

        assert model.coef_.shape == (len(Y.columns), len(X.columns))

        results = {}
        for i, col in enumerate(Y.columns):
            y = Y[col]
            y_pred = Y_hat[:, i]

            score = r2_score(y, y_pred)
            results[col] = {
                "r2": score,
                "coef": model.coef_[i],
                "intercept": model.intercept_[i],
            }

        results = pd.DataFrame.from_dict(
            results, orient="index", columns=["r2", "coef", "intercept"]
        )
        results_dict[feature_name] = results

    results_dict
    print("done")


if __name__ == "__main__":
    app.run(main)
