import datetime as dt
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from lobster_tools.flow import _FEATURE_NAMES_WITH_MARGINALS, evaluate_features

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

        # etf_executions = pd.read_pickle(etf_dir / "etf_executions.pkl")
        # tags = pd.read_pickle(lookback_dir / "tags.pkl")
        # TAG_COLUMNS = tags.columns

        # etf_executions = etf_executions.drop(columns="notional")
        # full = etf_executions.join(tags, how="inner")

        # ofi calculation
        # ofis_dict = {}
        # for col in TAG_COLUMNS:
        #     logging.info(f"computing ofi for {col}")
        #     ofis = full.groupby([full.index.date, full[col]], observed=False).apply(
        #         ofi, resample_freq=resample_freq
        #     )
        #     ofis.index = ofis.index.droplevel(0)
        #     ofis = ofis.unstack(level=0)
        #     ofis = ofis.fillna(0)
        #     ofis.columns = ofis.columns.tolist()
        #     ofis_dict[col] = ofis

        # # write to disk
        # resample_dir = lookback_dir / "resample_freq" / resample_freq
        # resample_dir.mkdir(parents=True, exist_ok=True)

        # print(ofis_dict)
        # pd.to_pickle(ofis_dict, resample_dir / "ofis.pkl")

        ################################################################################
        ## regressions
        ################################################################################
        ofi_dict = pd.read_pickle(lookback_dir / "resample_freq" / resample_freq / "ofis.pkl")

        resample_dir = etf_dir / "resample_freq" / resample_freq
        returns = pd.read_pickle(resample_dir / "returns.pkl")

        for feature, ofi_df in ofi_dict.items():
            results = []
            for tag, ofi_ser in ofi_df.items():
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
                        {"feature": feature, "tag": tag, "markout": markout, "score": score}
                    )

        results = pd.DataFrame(results)
        feature_dir = lookback_dir / "resample_freq" / resample_freq / "feature" / feature
        feature_dir.mkdir(parents=True, exist_ok=True)
        results.to_pickle(feature_dir / "results.pkl")

    return
    ##############################################################
    # moved to regressions.py
    ##############################################################

    # load returns data

    # univariate regressions
    for feature, ofi_df in ofis_dict.items():
        results = []
        for tag, ofi_ser in ofi_df.items():
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
