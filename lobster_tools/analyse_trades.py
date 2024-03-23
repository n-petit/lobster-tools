"""Find highest annualised sharpes."""

import itertools as it

import pandas as pd
from absl import app, flags, logging

import lobster_tools.config  # noqa: F401
from lobster_tools.utils import get_dir

FLAGS = flags.FLAGS


def main(_):
    logging.info(f"FLAGS: {FLAGS}")

    dfs = []
    for epsilon, resample_freq in it.product(FLAGS.epsilons, FLAGS.resample_freqs):
        dir_ = get_dir(
            etf=FLAGS.etf,
            epsilon=epsilon,
            quantiles=FLAGS.quantiles,
            lookback=FLAGS.lookback,
            resample_freq=resample_freq,
        )

        # dict of {feature_name: pnl_results_df}
        univariate_pnl_results = pd.read_pickle(dir_ / "univariate_pnl_results.pkl")
        df = pd.concat(
                results.assign(
                    feature=feature, epsilon=epsilon, resample_freq=resample_freq
                )
                for feature, results in univariate_pnl_results.items()
        ).reset_index(drop=True)

        dfs.append(df)

    all_results: pd.DataFrame = pd.concat(dfs).reset_index(drop=True)

    # find highest abs(annualised_sharpe) for future markouts
    all_results = all_results.query("markout != 'contemp'")
    TRADE_FREQ_THRESHOLD = 0.1  # filter for strategies with high enough trade frequency
    all_results = all_results.query(f"trade_freq >= {TRADE_FREQ_THRESHOLD}")
    all_results = all_results.reindex(
        all_results.annualised_sharpe.abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    # reorder columns
    cols = list(all_results.columns)
    cols = cols[-3:] + cols[:-3]
    all_results = all_results[cols]

    # write top 10 to csv
    dir_ = get_dir(etf=FLAGS.etf) / "plots"
    all_results.head(10).to_csv(dir_ / "highest_sharpes.csv", sep="\t")


if __name__ == "__main__":
    app.run(main)
