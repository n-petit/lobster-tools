import datetime as dt

import numpy as np
import pandas as pd
from absl import app, flags, logging

from lobster_tools import config  # noqa: F401
from lobster_tools.utils import get_dir

FLAGS = flags.FLAGS


def main(_):
    for epsilon in FLAGS.epsilons:
        dir_ = get_dir(etf=FLAGS.etf, epsilon=epsilon)

        features = pd.read_pickle(dir_ / "features.pkl")
        neighbors = pd.read_pickle(dir_ / "neighbors.pkl")
        quantiles = pd.read_pickle(dir_ / "quantiles.pkl")

        # select quantiles
        mask = quantiles.index.get_level_values("quantile").isin(FLAGS.quantiles)
        quantiles = quantiles[mask]

        # rolling quantiles
        rolling_quantiles = (
            quantiles.groupby(level="quantile").rolling(FLAGS.lookback).mean()
        )
        assert rolling_quantiles.index.names == ["quantile", "date", "quantile"]
        rolling_quantiles.index = rolling_quantiles.index.droplevel(-1)
        rolling_quantiles = rolling_quantiles.dropna()
        assert rolling_quantiles.ge(0).all().all()

        # restrict
        tmp = rolling_quantiles.index.get_level_values("date")
        first_date = tmp.unique().asof(tmp.min() + dt.timedelta(days=1))
        features = features[features.index.date >= first_date]

        # discrete tags
        mapping = {
            0: 0,
            1: 1,
            2: 1,
            3: 2,
            4: 2,
            5: 3,
            6: 3,
            7: 4,
            8: 4,
            # apply >= 9: 5 in lambda expression
        }
        _last_key_in_mapping = max(mapping.keys())
        _final_tag = max(mapping.values()) + 1

        # discrete tagging (numTrades and distinctTickers)
        discrete_features = features.select_dtypes(include="int").columns
        discrete_tags = features[discrete_features].map(
            lambda x: mapping[x] if x <= _last_key_in_mapping else _final_tag
        )

        # continuous features tags (notionalTraded)
        continuous_features = features.select_dtypes(include="float").columns
        features = features[continuous_features]

        cts_tags = pd.DataFrame(index=features.index, columns=continuous_features)

        for date, group in features.groupby(features.index.date):
            previous_trading_day = (
                rolling_quantiles.index.get_level_values("date")
                .unique()
                .asof(date - dt.timedelta(days=1))
            )
            quantiles_for_date = rolling_quantiles.xs(
                previous_trading_day, level="date"
            )

            for column in group.columns:
                bins = quantiles_for_date[column].values
                assert not np.isclose(bins[0], 0, atol=1e-2)
                print(column)
                print(bins)
                bins = np.concatenate(([-np.inf, 0.0], bins, [np.inf]))
                # features in data are positive, therefore (-inf, 0.0] == {0.0}

                cts_tags.loc[group.index, column] = pd.cut(
                    group[column],
                    bins=bins,
                    labels=range(
                        len(bins) - 1
                    ),  # tag 0 for (-inf, 0], 1 for (0, Q1] etc...
                    include_lowest=False,  # (-inf, 0.0], (0.0, Q1], (Q1, Q2], ...
                )

        cts_tags = cts_tags.astype("int")  # so that -1 for iso tag can be assigned

        # merge discrete tags and continuous tags
        assert cts_tags.index.equals(discrete_tags.index)
        tags = pd.concat((discrete_tags, tags), axis=1)

        # tags other than the quantiles
        _ISOLATED_TAG = -1
        neighbors = neighbors[
            neighbors.index.date >= first_date
        ]  # restrict to first date onwards
        tags = neighbors.join(tags, how="left")
        # tags = tags[tags.index.date >= first_date]  # restrict to first date onwards

        tag_cols = tags.drop(columns=neighbors.columns).columns
        tags.loc[~tags.nonIso, tag_cols] = _ISOLATED_TAG

        tags = tags.drop(columns=neighbors.columns)  # drop columns from neighbors

        # TODO: improve astypes or change the way that isolated tag is assigned
        tags = tags.astype("int").astype(
            "category"
        )  # cannot set as categorical before here

        # save to file
        dir_ = get_dir(
            etf=FLAGS.etf,
            epsilon=epsilon,
            quantiles=FLAGS.quantiles,
            lookback=FLAGS.lookback,
        )
        dir_.mkdir(parents=True, exist_ok=True)

        tags.to_pickle(dir_ / "tags.pkl")

    return


if __name__ == "__main__":
    app.run(main)
