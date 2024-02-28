import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from lobster_tools import config  # noqa: F401
from lobster_tools.config import FEATURE_NAMES_WITH_MARGINALS

# TODO: allow for (0.1, 0.9), (0.1, 0.5, 0.9) ...
flags.DEFINE_multi_float(
    "quantiles",
    [0.1, 0.9],
    "Quantiles for the features.",
    lower_bound=0.0,
    upper_bound=1.0,
)
flags.DEFINE_integer(
    "lookback",
    20,
    "Lookback period in days for rolling quantiles.",
)

FLAGS = flags.FLAGS


def main(_):
    # load data
    logging.info("loading data")
    output_dir = Path(FLAGS.output_dir)
    etf_dir = output_dir / "etf" / FLAGS.etf
    etf_executions = pd.read_pickle(etf_dir / "etf_executions.pkl")

    for epsilon in FLAGS.epsilons:
        eps_dir = etf_dir / "epsilon" / epsilon
        features = pd.read_pickle(eps_dir / "features.pkl")
        neighbors = pd.read_pickle(eps_dir / "neighbors.pkl")
        quantiles = pd.read_pickle(eps_dir / "quantiles.pkl")

        # select only quantiles at a given level
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


        # these will be numTrades and distinctTickers
        discrete_features = features.select_dtypes(include="int").columns
        discrete_tags = features[discrete_features].map(
            lambda x: mapping[x] if x <= _last_key_in_mapping else _final_tag
        )

        # continuous features tags
        continuous_features = features.select_dtypes(include="float").columns
        features = features[continuous_features]

        # _NOTIONAL_ONLY_FEATURES = [
        #     x for x in FEATURE_NAMES_WITH_MARGINALS if "notional" in x
        # ]
        # features = features[_NOTIONAL_ONLY_FEATURES]

        # NOTE: tags here is continuous tags
        tags = pd.DataFrame(index=features.index, columns=continuous_features)

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
                # (-inf, 0.0] == {0.0} since features >= 0

                tags.loc[group.index, column] = pd.cut(
                    group[column],
                    bins=bins,
                    labels=range(
                        len(bins) - 1
                    ),  # tag 0 for (-inf, 0], 1 for (0, Q1] etc...
                    include_lowest=False,  # (-inf, 0.0], (0.0, Q1], (Q1, Q2], ...
                )
        
        tags = tags.astype("int") # so that -1 for iso tag can be assigned

        # merge discrete tags and continuous tags
        assert tags.index.equals(discrete_tags.index)
        tags = pd.concat((discrete_tags, tags), axis=1)

        # tags other than the quantiles
        _ISOLATED_TAG = -1
        neighbors = neighbors[neighbors.index.date >= first_date]  # restrict to first date onwards
        tags = neighbors.join(tags, how="left")
        # tags = tags[tags.index.date >= first_date]  # restrict to first date onwards
        
        tag_cols = tags.drop(columns=neighbors.columns).columns
        tags.loc[~tags.nonIso, tag_cols] = _ISOLATED_TAG

        tags = tags.drop(columns=neighbors.columns) # drop columns from neighbors

        # TODO: improve astypes or change the way that isolated tag is assigned
        tags = tags.astype('int').astype('category') # cannot set as categorical before here

        # save to file
        quantiles_dir = eps_dir / "quantiles" / "_".join(map(str, FLAGS.quantiles))
        quantiles_dir.mkdir(parents=True, exist_ok=True)
        lookback_dir = quantiles_dir / "lookback" / str(FLAGS.lookback)
        lookback_dir.mkdir(parents=True, exist_ok=True)

        tags.to_pickle(lookback_dir / "tags.pkl")

    return

    ###########################################################
    ###########################################################
    # move to regressions.py file

    # don't really need full. so will redo above with neighbors
    full = pd.concat((etf_executions.drop(columns="notional"), neighbors), axis=1).join(
        tags, how="left"
    )
    full = full[full.index.date >= first_date]  # restrict to first date onwards
    full.loc[~full.nonIso, FEATURE_NAMES_WITH_MARGINALS] = _ADDITIONAL_TAGS["iso"]
    # convert floats to ints for nonIso
    full.loc[full.nonIso, FEATURE_NAMES_WITH_MARGINALS] = full.loc[
        full.nonIso, FEATURE_NAMES_WITH_MARGINALS
    ].astype("int")
    full = full.astype({col: "category" for col in FEATURE_NAMES_WITH_MARGINALS})

    # ofi calculation
    ofis_dict = {}
    for col in FEATURE_NAMES_WITH_MARGINALS:
        logging.info(f"computing ofi for {col}")

        ofis = full.groupby([full.index.date, full[col]], observed=False).apply(ofi)
        ofis.index = ofis.index.droplevel(0)
        ofis = ofis.unstack(level=0)
        ofis = ofis.fillna(0)
        ofis.columns = ofis.columns.tolist()
        ofis_dict[col] = ofis

    # load returns data
    resample_dir = etf_dir / "resample_freq" / FLAGS.resample_freq
    returns = pd.read_pickle(resample_dir / "returns.pkl")

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


if __name__ == "__main__":
    app.run(main)
