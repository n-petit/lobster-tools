import itertools as it
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from absl import app, flags, logging
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import lobster_tools.config  # noqa: F401
from lobster_tools import returns, tagging  # noqa: F401

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

    for epsilon, resample_freq in it.product(FLAGS.epsilons, FLAGS.resample_freqs):
        logging.info("flags.FLAGS: ", FLAGS)

        eps_dir = etf_dir / "epsilon" / epsilon
        quantile_dir = eps_dir / "quantiles" / "_".join(map(str, FLAGS.quantiles))
        lookback_dir = quantile_dir / "lookback" / str(FLAGS.lookback)
        lookback_resample_dir = lookback_dir / "resample_freq" / resample_freq

        # dict of {feature_name: df_coi_per_tag}
        cois_dict: dict[str, pd.DataFrame] = pd.read_pickle(
            lookback_resample_dir / "cois.pkl"
        )

        resample_dir = etf_dir / "resample_freq" / resample_freq
        returns = pd.read_pickle(resample_dir / "returns.pkl")

        def get_univariate_regression_results(coi: pd.DataFrame, returns: pd.DataFrame):
            """For a given feature, compute univariate regressions for each tag in `coi`
            and for each markout in `returns`."""
            results = []
            for tag, coi_ser in coi.items():
                coi_ser = coi_ser.dropna()
                for markout, return_ser in returns.items():
                    return_ser = return_ser.dropna()
                    X, Y = restrict_common_index(coi_ser, return_ser)
                    X = X.values.reshape(-1, 1)
                    # NOTE: need to be separate models due to different NaNs in returns
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
            return results

        univariate_regression_results_dict = {
            feature: get_univariate_regression_results(coi, returns)
            for feature, coi in cois_dict.items()
        }

        # no tagging
        coi_no_tagging = pd.read_pickle(
            etf_dir / "resample_freq" / resample_freq / "coi_no_tagging.pkl"
        )

        def get_untagged_regression_results(
            coi_no_tagging: pd.Series, returns: pd.DataFrame
        ):
            """Compute univariate regressions for each markout in `returns`. Used for
            a pd.Series of untagged COIs."""
            results = []
            for markout, return_ser in returns.items():
                return_ser = return_ser.dropna()
                X, Y = restrict_common_index(coi_no_tagging, return_ser)
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
            results = results.set_index("markout", drop=True)
            return results

        untagged_regression_results = get_untagged_regression_results(
            coi_no_tagging, returns
        )
        untagged_regression_results.to_pickle(
            etf_dir
            / "resample_freq"
            / resample_freq
            / "untagged_regression_results.pkl"
        )

        # increase in R2 score relative to COIs computed on untagged flow
        def assign_delta_score(
            results: pd.DataFrame, untagged_regression_results: pd.DataFrame
        ):
            results = results.merge(
                untagged_regression_results, on="markout", suffixes=("", "_untagged")
            )
            results["delta_score"] = results["score"] - results["score_untagged"]
            return results

        # with delta score assined
        univariate_regression_results_dict = {
            feature: assign_delta_score(
                univariate_regression_results, untagged_regression_results
            )
            for feature, univariate_regression_results in univariate_regression_results_dict.items()
        }

        pd.to_pickle(
            univariate_regression_results_dict,
            lookback_resample_dir / "univariate_regressions.pkl",
        )

        # TODO: move plots to separate file after?
        plot_dir = etf_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # def plot_delta_scores(feature, df: pd.DataFrame):
        #     p = sns.catplot(
        #         data=df, x="markout", y="delta_score", hue="tag", kind="bar"
        #     )
        #     p.set(title=feature)
        #     p.set_xticklabels(rotation=45)
        #     plt.savefig(plot_dir / f"{feature}.png")

        # for (
        #     feature,
        #     univariate_regression_results,
        # ) in univariate_regression_results_dict.items():
        #     plot_delta_scores(feature, univariate_regression_results)

        # for generating a single pdf with each page as
        # def plot_delta_scores(feature, df: pd.DataFrame):
        #     p = sns.catplot(
        #         data=df, x="markout", y="delta_score", hue="tag", kind="bar"
        #     )
        #     p.set(title=feature)
        #     p.set_xticklabels(rotation=45)
        #     plt.subplots_adjust(bottom=0.2, top=0.9) # title was being cutoff
        #     return p.figure

        # with PdfPages(plot_dir / 'all_plots.pdf') as pdf:
        #     for feature, univariate_regression_results in univariate_regression_results_dict.items():
        #         fig = plot_delta_scores(feature, univariate_regression_results)
        #         pdf.savefig(fig)
        #         plt.close(fig)

        # same as above but with subplots
        def plot_delta_scores(ax, feature, df: pd.DataFrame):

            # NOTE: not using this at the moment
            # plot pct increase in R2 scores for each tag
            # df = df.eval('pct_change_score = delta_score / score_untagged')
            # p = sns.barplot(
            #     data=df, x="markout", y="pct_change_score", hue="tag", ax=ax, palette="Dark2"
            # )

            # plot delta of R2 scores for each tag
            p = sns.barplot(
                data=df, x="markout", y="score", hue="tag", ax=ax, palette="Dark2"
            )
            
            # set title and rotate x-axis labels
            ax.set_title(feature)
            plt.setp(ax.get_xticklabels(), rotation=45)

        pdf_plot_nrows = 7
        pdf_plot_ncols = 4

        plot_name = f"score_etf_{FLAGS.etf}_resample_freq_{resample_freq}_epsilon_{epsilon}_lookback_{FLAGS.lookback}_quantiles_{FLAGS.quantiles}"
        with PdfPages(plot_dir / f"{plot_name}.pdf") as pdf:
            fig, axs = plt.subplots(pdf_plot_nrows, pdf_plot_ncols, figsize=(20, 30))
            axs = axs.flatten()
            for ax, (feature, univariate_regression_results) in zip(
                axs, univariate_regression_results_dict.items()
            ):
                plot_delta_scores(ax, feature, univariate_regression_results)

            # store info about plot store in last axs ojbect
            ax_title = axs[-1]
            ax_title.text(
                0.5,
                0.5,
                textwrap.dedent(\
                    f"""
                    ETF: {FLAGS.etf}
                    resample_freq: {resample_freq}
                    epsilon: {epsilon}
                    lookback: {FLAGS.lookback}
                    quantiles: {FLAGS.quantiles}
                    """),
                ha="center",
                va="center",
                fontsize=24,
            )
            ax_title.axis("off")

            # set layout and save
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return

    # TODO: plot multivariate also
    # TOOD: plot bars of univariate from the base R2 scores
    
    # multivariate regression
    logging.info("running multivariate regressions")

    def get_multivariate_regression_results(coi: pd.DataFrame, returns: pd.DataFrame):
        """For a given feature, compute multivariate regressions using all COIs for
        each tag to regress against a single markout."""
        results = []
        for markout, return_ser in returns.items():
            return_ser = return_ser.dropna()
            X, Y = restrict_common_index(coi, return_ser)

            # X = X.values.reshape(-1, 1) # no need to reshape for multivariate
            # NOTE: need to be separate models due to different NaNs in returns
            model = LinearRegression()
            model.fit(X, Y)
            Y_hat = model.predict(X)
            score = r2_score(Y, Y_hat)

            n, p = X.shape
            adj_score = 1 - (1 - score) * (n - 1) / (n - p - 1)

            results.append({"markout": markout, "score": score, "adj_score": adj_score})

        results = pd.DataFrame(results)
        results = results.set_index("markout", drop=True)
        return results

    multivariate_regression_results_dict = {
        feature: get_multivariate_regression_results(coi, returns)
        for feature, coi in cois_dict.items()
    }

    pd.to_pickle(
        multivariate_regression_results_dict,
        lookback_resample_dir / "multivariate_regressions.pkl",
    )

    return

    # still keeping here because of multivariate regression computation
    ##############################################################
    # moved to regressions.py
    ##############################################################

    # load returns data

    # univariate regressions
    for feature, coi_df in ofis_dict.items():
        results = []
        for tag, coi_no_tagging in coi_df.items():
            # TODO: only drop na if necessary
            coi_no_tagging = coi_no_tagging.dropna()
            returns = returns.dropna()
            X, Y = restrict_common_index(coi_no_tagging, returns)
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
    univariate_regression_results_dict = {}
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
        univariate_regression_results_dict[feature_name] = results

    univariate_regression_results_dict
    print("done")


if __name__ == "__main__":
    app.run(main)
