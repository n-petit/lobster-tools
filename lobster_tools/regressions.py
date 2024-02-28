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


def restrict_common_index(
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

    for epsilon, resample_freq in itertools.product(
        FLAGS.epsilons, FLAGS.resample_freqs
    ):
        logging.info("flags.FLAGS: ", FLAGS)

        eps_dir = etf_dir / "epsilon" / epsilon
        quantile_dir = eps_dir / "quantiles" / "_".join(map(str, FLAGS.quantiles))
        lookback_dir = quantile_dir / "lookback" / str(FLAGS.lookback)
        resample_dir2 = lookback_dir / "resample_freq" / resample_freq

        cois_dict = pd.read_pickle(
            resample_dir2 / "cois.pkl"
        )

        # TODO: better folder structure
        resample_dir = etf_dir / "resample_freq" / resample_freq
        returns = pd.read_pickle(resample_dir / "returns.pkl")

        res_dict = {}
        for feature, coi_df in cois_dict.items():
            results = []
            for tag, ofi_ser in coi_df.items():
                ofi_ser = ofi_ser.dropna()
                for markout, return_ser in returns.items():
                    return_ser = return_ser.dropna()
                    X, Y = restrict_common_index(ofi_ser, return_ser)
                    X = X.values.reshape(-1, 1)
                    model = LinearRegression()
                    model.fit(X, Y)
                    Y_hat = model.predict(X)
                    score = r2_score(Y, Y_hat)

                    results.append(
                        {
                            "tag": tag,
                            "markout": markout,
                            "score": score,
                        }
                    )
            results = pd.DataFrame(results)
            res_dict[feature] = results


        # feature_dir = (
        #     lookback_dir / "resample_freq" / resample_freq / "feature" / feature
        # )
        # feature_dir.mkdir(parents=True, exist_ok=True)
        # results.to_pickle(feature_dir / "results.pkl")


        plot_dir = etf_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        for feature, df in res_dict.items():
            p = sns.catplot(data=df, x='markout', y='score', hue='tag', kind='bar')
            p.set(title=feature)
            p.set_xticklabels(rotation=45)
            plt.savefig(plot_dir / f"{feature}_univariate_regressions.png")



        # no tagging
        ofi_ser = pd.read_pickle(etf_dir / "resample_freq" / resample_freq / "coi_no_tagging.pkl")
        results = []
        for markout, return_ser in returns.items():
            return_ser = return_ser.dropna()
            X, Y = restrict_common_index(ofi_ser, return_ser)
            X = X.values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, Y)
            Y_hat = model.predict(X)
            score = r2_score(Y, Y_hat)

            results.append(
                {
                    "markout": markout,
                    "score": score,
                }
            )
        results = pd.DataFrame(results)

        fig = plt.figure()
        sns.barplot(results, y='score', x='markout')
        plt.savefig(plot_dir / f"no_tagging_univariate_regressions.png")

    
    return
    ##############################################################
    # moved to regressions.py
    ##############################################################

    # load returns data

    # univariate regressions
    for feature, coi_df in ofis_dict.items():
        results = []
        for tag, ofi_ser in coi_df.items():
            # TODO: only drop na if necessary
            ofi_ser = ofi_ser.dropna()
            returns = returns.dropna()
            X, Y = restrict_common_index(ofi_ser, returns)
            X = X.values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, Y)
            Y_hat: np.ndarray = model.predict(X)

            for i, markout in enumerate(Y.columns):
                y = Y[markout]
                y_pred = Y_hat[:, i]
                score = r2_score(y, y_pred)
                results.append(
                    {"feature": feature, "tag": tag, "markout": markout, "score": score}
                )
        results = pd.DataFrame(results)
        sns.catplot(data=results, x="tag", y="score", hue="markout", kind="bar")

        feature_dir = lookback_dir / "feature" / feature
        feature_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(feature_dir / f"univariate_regressions.png")
    print("hi")
    ###########################################
    ###########################################
    return
    ###########################################
    ###########################################

    # multivariate regression
    logging.info("running multivariate regressions")
    results_dict = {}
    for feature_name, ofis in ofis_dict.items():
        ofis = ofis.dropna()
        returns = returns.dropna()

        X, Y = restrict_common_index(ofis, returns)
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
