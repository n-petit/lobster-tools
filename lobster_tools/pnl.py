"""Compute pnls and sharpes for the tagged and untagged flows."""

import itertools as it
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from absl import app, flags, logging
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import lobster_tools.config  # noqa: F401
from lobster_tools.utils import get_dir

FLAGS = flags.FLAGS

TRADE_THRESHOLDS = (0.0, 0.25, 0.5, 0.75, 0.99)


def restrict_common_index(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict two dataframes to their common index."""
    common_index = df1.index.intersection(df2.index)
    return df1.loc[common_index], df2.loc[common_index]


def main(_):
    logging.info(f"FLAGS: {FLAGS}")

    for epsilon, resample_freq in it.product(FLAGS.epsilons, FLAGS.resample_freqs):
        print("###### GREPGREP ######")
        print(f"epsilons: {epsilon}, resample_freq: {resample_freq}")

        # load data
        logging.info("loading data")

        dir_ = get_dir(
            etf=FLAGS.etf,
            epsilon=epsilon,
            quantiles=FLAGS.quantiles,
            lookback=FLAGS.lookback,
            resample_freq=resample_freq,
        )
        # dict of {feature_name: df_coi_per_tag}
        cois_dict: dict[str, pd.DataFrame] = pd.read_pickle(dir_ / "cois.pkl")

        dir_ = get_dir(
            etf=FLAGS.etf,
            resample_freq=resample_freq,
        )
        returns = pd.read_pickle(dir_ / "returns.pkl")

        def get_trades(coi: pd.Series, threshold: float) -> pd.Series:
            """Returns a series of trades (-1, 0, 1) based on value of COI. Trade in the
            same direction as the COI if abs(COI) >= threshold.
            """
            trades = np.sign(coi)
            trades = trades.where(abs(coi) > threshold, 0)
            trades = trades[trades != 0]

            return trades

        def get_pnls(
            cois: pd.DataFrame, returns: pd.DataFrame, thresholds
        ) -> pd.DataFrame:
            results = []
            for (tag, coi_ser), (markout, returns_ser), threshold in it.product(
                cois.items(), returns.items(), thresholds
            ):
                print(f"tag: {tag}, markout: {markout}, threshold: {threshold}")

                coi_ser = coi_ser.dropna()
                returns_ser = returns_ser.dropna()
                trades = get_trades(coi_ser, threshold)

                trades, returns_ser = restrict_common_index(trades, returns_ser)

                pnls = trades.mul(returns_ser)

                pnls_gp = pnls.groupby(pnls.index.date)
                total_pnl_per_day = pnls_gp.sum()
                average_pnl_per_day = pnls_gp.mean()
                trades_per_day = pnls_gp.count()

                # aggregate statistics
                if len(trades) == 0:
                    ppt_global = 0
                    annualised_sharpe = 0
                    trade_freq = 0
                else:
                    num_distinct_dates = len(pnls_gp)
                    annualised_sharpe = (
                        np.sqrt(num_distinct_dates)
                        * total_pnl_per_day.mean()
                        / total_pnl_per_day.std()
                    )

                    # pnl per trade global = (sum pnl for day) / (sum of bet sizes)
                    ppt_global = total_pnl_per_day.sum() / trades_per_day.sum()

                    # trade frequency
                    trade_freq = trades_per_day.sum() / len(coi_ser)

                # TODO: also check plots for ppt local
                # pnl per trade local = average (average pnl for day)
                # ppt_local = average_pnl_per_day.mean()

                results.append(
                    {
                        "tag": tag,
                        "markout": markout,
                        "threshold": threshold,
                        "annualised_sharpe": annualised_sharpe,
                        "ppt_global": ppt_global,
                        "trade_freq": trade_freq,
                    }
                )
            results = pd.DataFrame(results)
            results.fillna(0)
            return results

        univariate_pnl_results_dict = {
            feature: get_pnls(coi, returns, TRADE_THRESHOLDS)
            for feature, coi in cois_dict.items()
        }

        dir_ = get_dir(
            etf=FLAGS.etf,
            epsilon=epsilon,
            quantiles=FLAGS.quantiles,
            lookback=FLAGS.lookback,
            resample_freq=resample_freq,
        )
        pd.to_pickle(univariate_pnl_results_dict, dir_ / "univariate_pnl_results.pkl")

        # TODO: move plots to separate file
        # run plots
        def generate_plot(feature, results: pd.DataFrame):
            dir_ = get_dir(etf=FLAGS.etf) / "plots"
            dir_.mkdir(exist_ok=True)

            params = {
                "etf": FLAGS.etf,
                "feature": feature,
                "resampleFreq": resample_freq,
                "epsilon": epsilon,
                "lookback": FLAGS.lookback,
                "quantiles": FLAGS.quantiles,
            }
            plot_name = "_".join([f"{k}_{v}" for k, v in params.items()])
            plot_title = ", ".join([f"{k}:{v}" for k, v in params.items()])
            print(plot_title)

            with PdfPages(dir_ / f"{plot_name}.pdf") as pdf:
                stats_to_plot = ("annualised_sharpe", "ppt_global", "trade_freq")
                num_rows, num_cols = len(stats_to_plot), len(TRADE_THRESHOLDS)
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(30, 20))
                plt.subplots_adjust(hspace=0.5, wspace=0.5)
                plt.suptitle(plot_title, fontsize=24)

                for i, stat in enumerate(stats_to_plot):
                    for j, threshold in enumerate(TRADE_THRESHOLDS):
                        ax = axs[i, j]
                        df = results.query(f"threshold == {threshold}")
                        sns.barplot(
                            data=df,
                            x="markout",
                            y=stat,
                            hue="tag",
                            ax=ax,
                            palette="Dark2",
                        )
                        ax.set_title(f"threshold: {threshold}")
                        plt.setp(ax.get_xticklabels(), rotation=45)

                pdf.savefig(fig)
                plt.close(fig)
            return

        [
            generate_plot(feature, results)
            for feature, results in univariate_pnl_results_dict.items()
        ]

    return


#### comment start


#             # axs = axs.flatten()
#             for ax, (feature, univariate_regression_results) in zip(
#                 axs, univariate_pnl_results_dict.items()
#             ):
#                 # plot annualised sharpe


#                 stat_to_plot = ("annualised_sharpe", "ppt_global", "trade_freq")
#                 for i, stat in enumerate(stat_to_plot):
#                     sns.barplot(
#                         data=results, x="markout", y=stat, hue="tag", ax=ax, palette="Dark2"
#                     )
#                     ax.set_title(feature)
#                     plt.setp(ax.get_xticklabels(), rotation=45)


#             # store info about plot store in last axs ojbect
#             ax_title = axs[-1]
#             ax_title.text(
#                 0.5,
#                 0.5,
#                 textwrap.dedent(
#                     f"""
#                     ETF: {FLAGS.etf}
#                     resample_freq: {resample_freq}
#                     epsilon: {epsilon}
#                     lookback: {FLAGS.lookback}
#                     quantiles: {FLAGS.quantiles}
#                     """
#                 ),
#                 ha="center",
#                 va="center",
#                 fontsize=24,
#             )
#             ax_title.axis("off")

#             # set layout and save
#             plt.tight_layout()
#             pdf.savefig(fig)
#             plt.close(fig)

#     return
#     ##########################################################################
#     ##########################################################################
#     # old stuff

#     print("hi")
#     pd.to_pickle(
#         univariate_pnl_results_dict,
#         lookback_resample_dir / "univariate_pnl_results.pkl",
#     )

#     ################################## PLOTTING
#     # taken from regression.py

#     # same as above but with subplots
#     def plot_sharpes(ax, feature, df: pd.DataFrame):
#         p = sns.barplot(
#             data=df, x="markout", y="sharpe", hue="tag", ax=ax, palette="Dark2"
#         )

#         # set title and rotate x-axis labels
#         ax.set_title(feature)
#         plt.setp(ax.get_xticklabels(), rotation=45)

#     plot_dir = etf_dir / "plots"
#     plot_dir.mkdir(parents=True, exist_ok=True)

#     pdf_plot_nrows = 7
#     pdf_plot_ncols = 4

#     plot_name = f"sharpe_etf_{FLAGS.etf}_resample_freq_{resample_freq}_epsilon_{epsilon}_lookback_{FLAGS.lookback}_quantiles_{FLAGS.quantiles}"
#     with PdfPages(plot_dir / f"{plot_name}.pdf") as pdf:
#         fig, axs = plt.subplots(pdf_plot_nrows, pdf_plot_ncols, figsize=(20, 30))
#         axs = axs.flatten()
#         for ax, (feature, univariate_regression_results) in zip(
#             axs, univariate_pnl_results_dict.items()
#         ):
#             plot_sharpes(ax, feature, univariate_regression_results)

#         # store info about plot store in last axs ojbect
#         ax_title = axs[-1]
#         ax_title.text(
#             0.5,
#             0.5,
#             textwrap.dedent(
#                 f"""
#                 ETF: {FLAGS.etf}
#                 resample_freq: {resample_freq}
#                 epsilon: {epsilon}
#                 lookback: {FLAGS.lookback}
#                 quantiles: {FLAGS.quantiles}
#                 """
#             ),
#             ha="center",
#             va="center",
#             fontsize=24,
#         )
#         ax_title.axis("off")

#         # set layout and save
#         plt.tight_layout()
#         pdf.savefig(fig)
#         plt.close(fig)

#         # univariate_regression_results_dict = {
#         #     feature: get_univariate_regression_results(coi, returns)
#         #     for feature, coi in cois_dict.items()
#         # }
#     ################################## PLOTTING

#     continue
# return

# ####################################################################################
# def dummy_fn():  # to avoid indentation error
#     # no tagging
#     coi_no_tagging = pd.read_pickle(
#         etf_dir / "resample_freq" / resample_freq / "coi_no_tagging.pkl"
#     )

#     # place $1 bets based on ofi
#     long_short = coi_no_tagging.apply(np.sign)
#     pnl = returns_ser.mul(long_short, axis=0)

#     # plot cumsum across whole year and save plot
#     pnl.cumsum().plot()
#     plt.savefig(etf_dir / "plots" / "pnl.png")

#     # sharpe ratio for no tagging ofi strategy
#     sharpe = pnl.mean() / pnl.std()

#     ################################################################################
#     return
#     # stuff from regression file.

#     def get_univariate_regression_results(
#         coi_ser: pd.DataFrame, returns_ser: pd.DataFrame
#     ):
#         """For a given feature, compute univariate regressions for each tag in `coi`
#         and for each markout in `returns`."""
#         results = []
#         for tag, coi_ser in coi_ser.items():
#             coi_ser = coi_ser.dropna()
#             for markout, returns_ser in returns_ser.items():
#                 returns_ser = returns_ser.dropna()
#                 X, Y = restrict_common_index(coi_ser, returns_ser)
#                 X = X.values.reshape(-1, 1)
#                 # NOTE: need to be separate models due to different NaNs in returns
#                 model = LinearRegression()
#                 model.fit(X, Y)
#                 Y_hat = model.predict(X)
#                 score = r2_score(Y, Y_hat)

#                 results.append(
#                     {
#                         "tag": tag,
#                         "markout": markout,
#                         "score": score,
#                     }
#                 )
#         results = pd.DataFrame(results)
#         return results

#     univariate_regression_results_dict = {
#         feature: get_univariate_regression_results(coi_ser, returns_ser)
#         for feature, coi_ser in cois_dict.items()
#     }

#     # no tagging
#     coi_no_tagging = pd.read_pickle(
#         etf_dir / "resample_freq" / resample_freq / "coi_no_tagging.pkl"
#     )

#     def get_untagged_regression_results(
#         coi_no_tagging: pd.Series, returns_ser: pd.DataFrame
#     ):
#         """Compute univariate regressions for each markout in `returns`. Used for
#         a pd.Series of untagged COIs."""
#         results = []
#         for markout, returns_ser in returns_ser.items():
#             returns_ser = returns_ser.dropna()
#             X, Y = restrict_common_index(coi_no_tagging, returns_ser)
#             X = X.values.reshape(-1, 1)
#             model = LinearRegression()
#             model.fit(X, Y)
#             Y_hat = model.predict(X)
#             score = r2_score(Y, Y_hat)

#             results.append(
#                 {
#                     "markout": markout,
#                     "score": score,
#                 }
#             )
#         results = pd.DataFrame(results)
#         results = results.set_index("markout", drop=True)
#         return results

#     untagged_regression_results = get_untagged_regression_results(
#         coi_no_tagging, returns_ser
#     )
#     untagged_regression_results.to_pickle(
#         etf_dir
#         / "resample_freq"
#         / resample_freq
#         / "untagged_regression_results.pkl"
#     )

#     # increase in R2 score relative to COIs computed on untagged flow
#     def assign_delta_score(
#         results: pd.DataFrame, untagged_regression_results: pd.DataFrame
#     ):
#         results = results.merge(
#             untagged_regression_results, on="markout", suffixes=("", "_untagged")
#         )
#         results["delta_score"] = results["score"] - results["score_untagged"]
#         return results

#     # with delta score assined
#     univariate_regression_results_dict = {
#         feature: assign_delta_score(
#             univariate_regression_results, untagged_regression_results
#         )
#         for feature, univariate_regression_results in univariate_regression_results_dict.items()
#     }

#     pd.to_pickle(
#         univariate_regression_results_dict,
#         lookback_resample_dir / "univariate_regressions.pkl",
#     )

#     # TODO: move plots to separate file after?
#     plot_dir = etf_dir / "plots"
#     plot_dir.mkdir(parents=True, exist_ok=True)

#     # def plot_delta_scores(feature, df: pd.DataFrame):
#     #     p = sns.catplot(
#     #         data=df, x="markout", y="delta_score", hue="tag", kind="bar"
#     #     )
#     #     p.set(title=feature)
#     #     p.set_xticklabels(rotation=45)
#     #     plt.savefig(plot_dir / f"{feature}.png")

#     # for (
#     #     feature,
#     #     univariate_regression_results,
#     # ) in univariate_regression_results_dict.items():
#     #     plot_delta_scores(feature, univariate_regression_results)

# # for generating a single pdf with each page as
# # def plot_delta_scores(feature, df: pd.DataFrame):
# #     p = sns.catplot(
# #         data=df, x="markout", y="delta_score", hue="tag", kind="bar"
# #     )
# #     p.set(title=feature)
# #     p.set_xticklabels(rotation=45)
# #     plt.subplots_adjust(bottom=0.2, top=0.9) # title was being cutoff
# #     return p.figure

# # with PdfPages(plot_dir / 'all_plots.pdf') as pdf:
# #     for feature, univariate_regression_results in univariate_regression_results_dict.items():
# #         fig = plot_delta_scores(feature, univariate_regression_results)
# #         pdf.savefig(fig)
# #         plt.close(fig)

# # same as above but with subplots
# def plot_sharpes(ax, feature, df: pd.DataFrame):

#     # NOTE: not using this at the moment
#     # plot pct increase in R2 scores for each tag
#     # df = df.eval('pct_change_score = delta_score / score_untagged')
#     # p = sns.barplot(
#     #     data=df, x="markout", y="pct_change_score", hue="tag", ax=ax, palette="Dark2"
#     # )

#     # plot delta of R2 scores for each tag
#     p = sns.barplot(
#         data=df, x="markout", y="delta_score", hue="tag", ax=ax, palette="Dark2"
#     )

#     # set title and rotate x-axis labels
#     ax.set_title(feature)
#     plt.setp(ax.get_xticklabels(), rotation=45)

# pdf_plot_nrows = 7
# pdf_plot_ncols = 4
# with PdfPages(plot_dir / "all_plots.pdf") as pdf:
#     fig, axs = plt.subplots(pdf_plot_nrows, pdf_plot_ncols, figsize=(20, 30))
#     axs = axs.flatten()
#     for ax, (feature, univariate_regression_results) in zip(
#         axs, univariate_regression_results_dict.items()
#     ):
#         plot_sharpes(ax, feature, univariate_regression_results)

#     # store info about plot store in last axs ojbect
#     ax_title = axs[-1]
#     ax_title.text(
#         0.5,
#         0.5,
#         textwrap.dedent(
#             f"""
#             ETF: {FLAGS.etf}
#             resample_freq: {FLAGS.resample_freqs}
#             epsilon: {FLAGS.epsilons}
#             lookback: {FLAGS.lookback}
#             quantiles: {FLAGS.quantiles}
#             """
#         ),
#         ha="center",
#         va="center",
#         fontsize=24,
#     )
#     ax_title.axis("off")

#     # set layout and save
#     plt.tight_layout()
#     pdf.savefig(fig)
#     plt.close(fig)

# # TODO: plot multivariate also
# # TOOD: plot bars of univariate from the base R2 scores

# # multivariate regression
# logging.info("running multivariate regressions")

# def get_multivariate_regression_results(
#     coi_ser: pd.DataFrame, returns_ser: pd.DataFrame
# ):
#     """For a given feature, compute multivariate regressions using all COIs for
#     each tag to regress against a single markout."""
#     results = []
#     for markout, returns_ser in returns_ser.items():
#         returns_ser = returns_ser.dropna()
#         X, Y = restrict_common_index(coi_ser, returns_ser)

#         # X = X.values.reshape(-1, 1) # no need to reshape for multivariate
#         # NOTE: need to be separate models due to different NaNs in returns
#         model = LinearRegression()
#         model.fit(X, Y)
#         Y_hat = model.predict(X)
#         score = r2_score(Y, Y_hat)

#         n, p = X.shape
#         adj_score = 1 - (1 - score) * (n - 1) / (n - p - 1)

#         results.append({"markout": markout, "score": score, "adj_score": adj_score})

#     results = pd.DataFrame(results)
#     results = results.set_index("markout", drop=True)
#     return results

# multivariate_regression_results_dict = {
#     feature: get_multivariate_regression_results(coi_ser, returns_ser)
#     for feature, coi_ser in cois_dict.items()
# }

# pd.to_pickle(
#     multivariate_regression_results_dict,
#     lookback_resample_dir / "multivariate_regressions.pkl",
# )

# return

# # still keeping here because of multivariate regression computation
# ##############################################################
# # moved to regressions.py
# ##############################################################

# # load returns data

# # univariate regressions
# for feature, coi_df in ofis_dict.items():
#     results = []
#     for tag, coi_no_tagging in coi_df.items():
#         # TODO: only drop na if necessary
#         coi_no_tagging = coi_no_tagging.dropna()
#         returns_ser = returns_ser.dropna()
#         X, Y = restrict_common_index(coi_no_tagging, returns_ser)
#         X = X.values.reshape(-1, 1)
#         model = LinearRegression()
#         model.fit(X, Y)
#         Y_hat: np.ndarray = model.predict(X)

#         for i, markout in enumerate(Y.columns):
#             y = Y[markout]
#             y_pred = Y_hat[:, i]
#             score = r2_score(y, y_pred)
#             results.append(
#                 {"feature": feature, "tag": tag, "markout": markout, "score": score}
#             )
#     results = pd.DataFrame(results)
#     sns.catplot(data=results, x="tag", y="score", hue="markout", kind="bar")

#     feature_dir = lookback_dir / "feature" / feature
#     feature_dir.mkdir(parents=True, exist_ok=True)

#     plt.savefig(feature_dir / f"univariate_regressions.png")
# print("hi")
# ###########################################
# ###########################################
# return
# ###########################################
# ###########################################

# # multivariate regression
# logging.info("running multivariate regressions")
# univariate_regression_results_dict = {}
# for feature_name, ofis in ofis_dict.items():
#     ofis = ofis.dropna()
#     returns_ser = returns_ser.dropna()

#     X, Y = restrict_common_index(ofis, returns_ser)
#     model = LinearRegression()
#     model.fit(X, Y)
#     Y_hat: np.ndarray = model.predict(X)

#     assert model.coef_.shape == (len(Y.columns), len(X.columns))

#     results = {}
#     for i, col in enumerate(Y.columns):
#         y = Y[col]
#         y_pred = Y_hat[:, i]

#         score = r2_score(y, y_pred)
#         results[col] = {
#             "r2": score,
#             "coef": model.coef_[i],
#             "intercept": model.intercept_[i],
#         }

#     results = pd.DataFrame.from_dict(
#         results, orient="index", columns=["r2", "coef", "intercept"]
#     )
#     univariate_regression_results_dict[feature_name] = results

# univariate_regression_results_dict
# print("done")


# coment end


if __name__ == "__main__":
    app.run(main)
