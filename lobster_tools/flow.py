import datetime as dt
import itertools as it
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging
from arcticdb import Arctic, QueryBuilder
from sklearn.neighbors import KDTree

from lobster_tools.preprocessing import Event, EventGroup

flags.DEFINE_list(
    "tolerances",
    ["40us", "200us", "1ms", "5ms", "25ms", "125ms"],
    "Epsilon neighbors in time.",
)
flags.DEFINE_string(
    "etf",
    "XLC",
    "ETF.",
)
flags.DEFINE_list(
    "date_range",
    None,
    "Date range.",
)

FLAGS = flags.FLAGS

ETF_TO_EQUITIES = {"SPY": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "BRK.B", "META", "TSLA", "UNH", "XOM", "JNJ", "JPM", "V", "PG", "LLY", "MA", "AVGO", "HD", "MRK", "CVX", "PEP", "ABBV", "KO", "COST", "PFE", "CRM", "MCD", "WMT", "TMO", "CSCO", "BAC", "AMD", "ACN", "ADBE", "ABT", "LIN", "CMCSA", "DIS", "ORCL", "NFLX", "WFC", "TXN", "DHR", "VZ", "NEE", "PM", "BMY", "RTX", "NKE", "HON", "UPS", "COP", "LOW", "UNP", "SPGI", "INTU", "AMGN", "QCOM", "IBM", "INTC", "SBUX", "BA", "PLD", "MDT", "GE", "AMAT", "GS", "CAT", "MS", "T", "NOW", "ELV", "ISRG", "MDLZ", "LMT", "BKNG", "BLK", "GILD", "DE", "SYK", "AXP", "TJX", "ADI", "ADP", "CVS", "MMC", "C", "VRTX", "AMT", "SCHW", "LRCX", "TMUS", "MO", "CB", "REGN", "ZTS", "MU", "SO", "PGR", "CI", "BSX", "FISV", "ETN", "BDX", "DUK", "PYPL", "SNPS", "EQIX", "CSX", "EOG", "TGT", "SLB", "AON", "CL", "CME", "HUM", "NOC", "ITW", "CDNS", "APD", "KLAC", "WM", "ICE", "ORLY", "CMG", "HCA", "ATVI", "MCK", "MMM", "SHW", "FDX", "EW", "GIS", "MPC", "PXD", "MCO", "CCI", "NSC", "FCX", "PNC", "ROP", "MSI", "KMB", "AZO", "MAR", "GD", "DG", "GM", "EMR", "SRE", "PSA", "F", "PSX", "NXPI", "EL", "DXCM", "APH", "MNST", "VLO", "FTNT", "AJG", "BIIB", "OXY", "ADSK", "USB", "AEP", "D", "PH", "JCI", "MRNA", "ECL", "TDG", "MCHP", "TFC", "ADM", "TRV", "CTAS", "AIG", "EXC", "CTVA", "ANET", "TT", "HSY", "COF", "IDXX", "TEL", "STZ", "CPRT", "HLT", "MSCI", "PCAR", "IQV", "O", "YUM", "AFL", "HES", "SYY", "DOW", "ON", "A", "WMB", "ROST", "XEL", "CNC", "WELL", "MET", "PAYX", "CARR", "VRSK", "NUE", "OTIS", "CHTR", "LHX", "AME", "DHI", "SPG", "ED", "EA", "NEM", "AMP", "KMI", "KR", "RMD", "CTSH", "FIS", "CSGP", "ROK", "DVN", "PPG", "FAST", "DD", "ILMN", "VICI", "KHC", "PEG", "CMI", "GWW", "BK", "PRU", "ALL", "MTD", "RSG", "GEHC", "DLTR", "KEYS", "ODFL", "BKR", "LEN", "ABC", "AWK", "HAL", "WEC", "CEG", "ZBH", "ACGL", "HPQ", "ANSS", "KDP", "DFS", "IT", "PCG", "DLR", "GPN", "VMC", "OKE", "EFX", "WST", "EIX", "ULTA", "MLM", "PWR", "ES", "WBD", "APTV", "FANG", "ALB", "SBAC", "AVB", "CBRE", "STT", "EBAY", "GLW", "URI", "TSCO", "XYL", "WTW", "TROW", "IR", "CDW", "FTV", "DAL", "CHD", "GPC", "ENPH", "LYB", "MPWR", "MKC", "CAH", "HIG", "TTWO", "WBA", "WY", "AEE", "BAX", "DTE", "VRSN", "MTB", "ALGN", "EQR", "FE", "STE", "IFF", "FSLR", "ETR", "CTRA", "DRI", "HOLX", "CLX", "EXR", "FICO", "PODD", "PPL", "INVH", "DOV", "HPE", "LH", "TDY", "COO", "LVS", "EXPD", "OMC", "NDAQ", "RJF", "CNP", "ARE", "BR", "K", "LUV", "FITB", "FLT", "VTR", "NVR", "RCL", "WAB", "MAA", "BALL", "CMS", "SEDG", "CAG", "ATO", "RF", "TYL", "GRMN", "HWM", "SWKS", "MOH", "SJM", "STLD", "IRM", "TRGP", "CINF", "LW", "UAL", "WAT", "PFG", "TER", "IEX", "PHM", "NTRS", "NTAP", "HBAN", "BRO", "MRO", "TSN", "FDS", "DGX", "RVTY", "AMCR", "EPAM", "IPG", "J", "EXPE", "JBHT", "RE", "CBOE", "AKAM", "BG", "BBY", "PTC", "LKQ", "SNA", "PAYC", "AVY", "ZBRA", "AES", "EQT", "ESS", "EVRG", "TXT", "CFG", "SYF", "AXON", "FMC", "TECH", "LNT", "POOL", "MGM", "CF", "WDC", "HST", "PKG", "UDR", "CHRW", "STX", "NDSN", "INCY", "MOS", "LYV", "TRMB", "KMX", "SWK", "WRB", "TAP", "CPT", "MAS", "BWA", "L", "CCL", "BF.B", "IP", "HRL", "VTRS", "TFX", "KIM", "NI", "DPZ", "APA", "ETSY", "JKHY", "LDOS", "WYNN", "PEAK", "CE", "CPB", "MKTX", "HSIC", "CRL", "TPR", "EMN", "GEN", "JNPR", "GL", "QRVO", "MTCH", "CDAY", "AAL", "PNR", "ALLE", "KEY", "FOXA", "ROL", "CZR", "FFIV", "PNW", "REG", "AOS", "BBWI", "UHS", "XRAY", "BIO", "HII", "NRG", "HAS", "RHI", "GNRC", "WHR", "NWSA", "PARA", "WRK", "BEN", "AAP", "BXP", "IVZ", "CTLT", "AIZ", "FRT", "NCLH", "SEE", "VFC", "ALK", "DXC", "DVA", "CMA", "OGN", "MHK", "RL", "ZION", "FOX", "LNC", "NWL", "NWS", "DISH", "VNT"], "XLF": ["BRK.B", "JPM", "V", "MA", "BAC", "WFC", "SPGI", "GS", "MS", "BLK", "AXP", "MMC", "C", "SCHW", "CB", "PGR", "FISV", "PYPL", "AON", "CME", "ICE", "MCO", "PNC", "AJG", "USB", "TFC", "TRV", "AIG", "COF", "MSCI", "AFL", "MET", "AMP", "FIS", "BK", "PRU", "ALL", "ACGL", "DFS", "GPN", "STT", "WTW", "TROW", "HIG", "MTB", "NDAQ", "RJF", "FLT", "FITB", "RF", "CINF", "PFG", "RE", "HBAN", "NTRS", "BRO", "FDS", "CBOE", "CFG", "SYF", "WRB", "L", "JKHY", "MKTX", "GL", "KEY", "BEN", "IVZ", "AIZ", "CMA", "ZION", "LNC"], "XLB": ["LIN", "APD", "SHW", "FCX", "ECL", "CTVA", "DOW", "NUE", "NEM", "PPG", "DD", "VMC", "MLM", "ALB", "LYB", "IFF", "BALL", "STLD", "AMCR", "AVY", "FMC", "CF", "PKG", "MOS", "IP", "CE", "EMN", "WRK", "SEE"], "XLK": ["MSFT", "AAPL", "NVDA", "AVGO", "CRM", "CSCO", "AMD", "ACN", "ADBE", "ORCL", "TXN", "INTU", "QCOM", "IBM", "INTC", "AMAT", "NOW", "ADI", "LRCX", "MU", "SNPS", "CDNS", "KLAC", "ROP", "MSI", "NXPI", "APH", "FTNT", "ADSK", "MCHP", "ANET", "TEL", "ON", "CTSH", "KEYS", "IT", "HPQ", "ANSS", "GLW", "CDW", "ENPH", "MPWR", "VRSN", "FSLR", "FICO", "HPE", "TDY", "SEDG", "TYL", "SWKS", "TER", "NTAP", "EPAM", "AKAM", "PTC", "ZBRA", "WDC", "STX", "TRMB", "JNPR", "GEN", "QRVO", "FFIV", "DXC"], "XLV": ["UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR", "BMY", "AMGN", "MDT", "ELV", "ISRG", "GILD", "SYK", "CVS", "VRTX", "REGN", "ZTS", "BSX", "CI", "BDX", "HUM", "HCA", "MCK", "EW", "DXCM", "BIIB", "MRNA", "IDXX", "IQV", "A", "CNC", "RMD", "ILMN", "MTD", "GEHC", "ABC", "ZBH", "WST", "CAH", "BAX", "ALGN", "STE", "HOLX", "PODD", "LH", "COO", "MOH", "WAT", "DGX", "RVTY", "TECH", "INCY", "TFX", "VTRS", "HSIC", "CRL", "UHS", "BIO", "XRAY", "CTLT", "DVA", "OGN"], "XLI": ["RTX", "HON", "UPS", "UNP", "BA", "GE", "CAT", "LMT", "DE", "ADP", "ETN", "CSX", "NOC", "ITW", "WM", "MMM", "FDX", "NSC", "GD", "EMR", "PH", "JCI", "TDG", "CTAS", "TT", "CPRT", "PCAR", "PAYX", "CARR", "VRSK", "OTIS", "LHX", "AME", "CSGP", "ROK", "FAST", "CMI", "GWW", "RSG", "ODFL", "EFX", "PWR", "URI", "XYL", "IR", "FTV", "DAL", "DOV", "EXPD", "BR", "LUV", "WAB", "HWM", "UAL", "IEX", "J", "JBHT", "SNA", "PAYC", "AXON", "TXT", "CHRW", "NDSN", "SWK", "MAS", "LDOS", "CDAY", "PNR", "AAL", "ALLE", "ROL", "AOS", "HII", "GNRC", "RHI", "ALK", "GEHC"], "XLU": ["NEE", "SO", "DUK", "SRE", "AEP", "D", "EXC", "XEL", "ED", "PEG", "AWK", "WEC", "CEG", "PCG", "EIX", "ES", "AEE", "DTE", "FE", "ETR", "PPL", "CNP", "CMS", "ATO", "AES", "EVRG", "LNT", "NI", "PNW", "NRG"], "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX", "ORLY", "CMG", "MAR", "AZO", "GM", "F", "HLT", "YUM", "ROST", "DHI", "LEN", "ULTA", "APTV", "EBAY", "TSCO", "GPC", "DRI", "LVS", "RCL", "NVR", "GRMN", "PHM", "EXPE", "BBY", "LKQ", "POOL", "MGM", "KMX", "CCL", "BWA", "ETSY", "DPZ", "WYNN", "TPR", "CZR", "BBWI", "HAS", "WHR", "AAP", "NCLH", "VFC", "MHK", "RL", "NWL"], "XLP": ["PG", "PEP", "KO", "COST", "MDLZ", "WMT", "PM", "MO", "TGT", "CL", "GIS", "KMB", "DG", "EL", "MNST", "ADM", "HSY", "STZ", "SYY", "KR", "KHC", "DLTR", "KDP", "CHD", "MKC", "WBA", "CLX", "K", "CAG", "SJM", "LW", "TSN", "BG", "TAP", "BF.B", "HRL", "CPB"], "XLE": ["XOM", "CVX", "EOG", "COP", "SLB", "MPC", "PXD", "PSX", "VLO", "OXY", "HES", "WMB", "KMI", "DVN", "BKR", "HAL", "OKE", "FANG", "CTRA", "TRGP", "MRO", "EQT", "APA"], "XLC": ["META", "GOOGL", "GOOG", "NFLX", "CMCSA", "ATVI", "TMUS", "VZ", "DIS", "CHTR", "EA", "T", "WBD", "TTWO", "OMC", "IPG", "LYV", "MTCH", "FOXA", "PARA", "NWSA", "FOX", "NWS", "DISH"], "IYR": ["PLD", "AMT", "EQIX", "CCI", "PSA", "O", "WELL", "SPG", "CSGP", "VICI", "DLR", "SBAC", "AVB", "CBRE", "WY", "EQR", "EXR", "INVH", "ARE", "VTR", "MAA", "SUI", "IRM", "WPC", "ESS", "UDR", "GLPI", "HST", "CPT", "KIM", "ELS", "LSI", "PEAK", "AMH", "REXR", "CUBE", "NLY", "REG", "LAMR", "COLD", "NNN", "Z", "EGP", "FR", "HR", "JLL", "BXP", "OHI", "FRT", "STAG", "BRX", "ADC", "STWD", "SRC", "AGNC", "AIRC", "MPW", "RYN", "RITM", "PCH", "BXMT", "NSA", "DOC", "CUZ", "KRC", "LXP", "HHC", "ZG", "OFC", "SBRA", "XTSLA", "EQC", "NHI", "VNO", "HIW", "DEI", "SLG", "JBGS", "OPEN", "MLPFT", "MARGIN_USD", "USD"]}  # fmt: skip

_COLUMNS = [
    "time",
    "event",
    "order_id",
    "size",
    "price",
    "direction",
    "ask_price_1",
    "bid_price_1",
]

_OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def aggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    duplicates = df[df.duplicated(subset=["datetime", "direction"], keep=False)]

    aggregated_sizes = duplicates.groupby(["datetime", "direction"])["size"].transform(
        "sum"
    )

    df.loc[duplicates.index, "size"] = aggregated_sizes
    _MERGED_ORDER_ID = -2
    df.loc[duplicates.index, ["event", "order_id"]] = [
        Event.AGGREGATED.value,
        _MERGED_ORDER_ID,
    ]
    df = df.drop_duplicates(subset=["datetime"], keep="last")
    df = df.set_index("datetime", drop=True)
    return df


_FEATURE_FUNCTIONS = ["notional", "numTrades", "distinctTickers"]
_SAME_SIGN_OPPOSITE_SIGN = ["ss", "os"]
_BEFORE_AFTER = ["bf", "af"]


def _combine_features(*args):
    return ["_".join(x) for x in it.product(*args)]


_FEATURE_NAMES = _combine_features(
    _FEATURE_FUNCTIONS, _SAME_SIGN_OPPOSITE_SIGN, _BEFORE_AFTER
)
_MARGINAL_BEFORE_AFTER = _combine_features(_FEATURE_FUNCTIONS, _SAME_SIGN_OPPOSITE_SIGN)
_MARGINAL_SAME_SIGN_OPPOSITE_SIGN = _combine_features(_FEATURE_FUNCTIONS, _BEFORE_AFTER)

_FEATURE_NAMES_WITH_MARGINALS = (
    _FEATURE_NAMES
    + _FEATURE_FUNCTIONS
    + _MARGINAL_BEFORE_AFTER
    + _MARGINAL_SAME_SIGN_OPPOSITE_SIGN
)


@dataclass
class Epsilon:
    _str: str
    timedelta: dt.timedelta = field(init=False)
    nanoseconds: int = field(init=False)

    def __post_init__(self):
        self.timedelta: pd.Timedelta = pd.Timedelta(self._str)
        self.nanoseconds = int(self.timedelta / pd.Timedelta(1, unit="ns"))

    def __str__(self) -> str:
        return self._str


def evaluate_features(
    equity_executions: pd.DataFrame,
    neighbors: np.ndarray,
    etf_trade_time: float,
    etf_trade_direction: int,
):
    cols = ["time", "size", "price", "direction", "ticker", "notional"]
    features = (
        equity_executions.iloc[neighbors][cols]
        .assign(
            equityBefore=lambda _df: _df.time < etf_trade_time,
            sameSign=lambda _df: _df.direction == etf_trade_direction,
        )
        .groupby(["sameSign", "equityBefore"])
        .agg(
            notional=("notional", "sum"),
            numTrades=("size", "count"),
            distinctTickers=("ticker", "nunique"),
        )
        .stack()
        .reorder_levels([-1, 0, 1])
    )

    sameSignOppositeSign = features.index.levels[1].map({True: "ss", False: "os"})
    equityBeforeAfter = features.index.levels[2].map({True: "bf", False: "af"})

    features.index = features.index.set_levels(
        levels=[sameSignOppositeSign, equityBeforeAfter], level=[1, 2]
    )
    features.index = features.index.to_flat_index()
    features.index = ["_".join(x) for x in features.index]

    features.reindex(_FEATURE_NAMES, fill_value=0.0)
    return features


def add_marginals(df: pd.DataFrame) -> pd.DataFrame:
    for col in _FEATURE_FUNCTIONS:
        # marginalise over bf/af and ss/os
        df[col] = (
            df[col + "_ss_af"]
            + df[col + "_ss_bf"]
            + df[col + "_os_af"]
            + df[col + "_os_bf"]
        )

        # marginalise over bf/af
        df[col + "_ss"] = df[col + "_ss_af"] + df[col + "_ss_bf"]
        df[col + "_os"] = df[col + "_os_af"] + df[col + "_os_bf"]

        # marginalise over ss/os
        df[col + "_af"] = df[col + "_ss_af"] + df[col + "_os_af"]
        df[col + "_bf"] = df[col + "_ss_bf"] + df[col + "_os_bf"]
    return df


def main(_):
    # set output directory and logging
    now = dt.datetime.now()
    etf_dir = _OUTPUT_DIR / f"{now:%Y-%m-%d}" / FLAGS.etf
    etf_dir.mkdir(parents=True)

    logging.get_absl_handler().use_absl_log_file(log_dir=etf_dir)

    # parse flags
    logging.info(f"FLAGS={FLAGS}")

    tolerances = [Epsilon(tolerance) for tolerance in FLAGS.tolerances]
    date_range = (
        tuple(
            dt.datetime.strptime(date, "%Y-%m-%d").date() for date in FLAGS.date_range
        )
        if FLAGS.date_range
        else None
    )
    etf = FLAGS.etf
    equities = ETF_TO_EQUITIES[etf]

    arctic = Arctic(FLAGS.s3_uri)
    arctic_library = arctic[FLAGS.library]

    # restrict to equities present in the library
    equities = [x for x in equities if x in arctic_library.list_symbols()]

    # fetch data
    q = QueryBuilder()
    q = q[q.event.isin(EventGroup.EXECUTIONS.value)]

    logging.info(f"Fetching data for ticker={equities}")
    equity_executions = (
        pd.concat(
            (
                arctic_library.read(
                    symbol=ticker,
                    columns=_COLUMNS,
                    query_builder=q,
                    date_range=date_range,
                )
                .data.pipe(aggregate_duplicates)
                .assign(
                    ticker=ticker,
                )
                .eval("mid = (bid_price_1 + ask_price_1) / 2")
                .eval("notional = price * size")
                for ticker in equities
            )
        )
        .sort_index()
        .astype({"ticker": "category"})
    )

    logging.info(f"Fetching data for ticker={etf}")
    etf_executions = (
        arctic_library.read(
            symbol=etf,
            query_builder=q,
            columns=_COLUMNS,
            date_range=date_range,
        )
        .data.pipe(aggregate_duplicates)
        .eval("mid = (bid_price_1 + ask_price_1) / 2")
        .eval("notional = price * size")
    )

    assert etf_executions.index.is_monotonic_increasing
    assert equity_executions.index.is_monotonic_increasing
    assert etf_executions.index.is_unique
    assert equity_executions.index.is_unique

    # save trade data to disk
    equity_executions.to_pickle(etf_dir / "equity_executions.pkl")
    etf_executions.to_pickle(etf_dir / "etf_executions.pkl")

    # compute kd tree
    etf_times = etf_executions.index.values.reshape(-1, 1)
    equity_times = equity_executions.index.values.reshape(-1, 1)

    kd_tree = KDTree(equity_times, metric="l1")

    # compute features
    for tolerance in tolerances:
        logging.info(f"Computing features for tolerance={str(tolerance)}")

        neighbors = pd.DataFrame(
            index=etf_executions.index, columns=["neighbors", "nonIso"]
        )
        neighbors["neighbors"] = kd_tree.query_radius(
            etf_times, r=tolerance.nanoseconds
        )
        neighbors["nonIso"] = neighbors["neighbors"].apply(lambda x: x.size > 0)

        features = pd.concat((etf_executions, neighbors), axis=1)[
            neighbors.nonIso
        ].apply(
            lambda row: evaluate_features(
                equity_executions=equity_executions,
                neighbors=row.neighbors,
                etf_trade_direction=row.direction,
                etf_trade_time=row.time,
            ),
            axis=1,
            result_type="expand",
        )

        features = features.fillna(0.0)

        feature_to_type = {
            "numTrades": "int",
            "distinctTickers": "int",
            "notional": "float",
        }
        for feature, dtype in feature_to_type.items():
            cols = features.filter(like=feature).columns
            features[cols] = features[cols].astype(dtype)

        features = add_marginals(features)

        # TODO: change tol_dir to be in directory called epsilon. Manually changed this
        # save to disk
        tol_dir = etf_dir / str(tolerance)
        tol_dir.mkdir()

        neighbors.to_pickle(tol_dir / "neighbors.pkl")
        features.to_pickle(tol_dir / "features.pkl")


if __name__ == "__main__":
    app.run(main)
