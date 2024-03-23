import itertools as it
from pathlib import Path

from absl import flags

base_dir = Path(__file__).parent.parent
output_dir = base_dir / "outputs"

flags.DEFINE_string(
    "output_dir",
    str(output_dir),
    "Output directory.",
)
flags.DEFINE_string(
    "s3_uri",
    "s3://10.146.104.101:9100:lobster?access=minioadmin&secret=minioadmin",
    "URI of the S3 bucket to connect to.",
)
flags.DEFINE_string(
    "library",
    "2021",
    "Arctic library to connect to.",
)
flags.DEFINE_string(
    "etf",
    "XLC",
    "ETF.",
)
flags.DEFINE_list(
    "epsilons",
    ["40us", "200us", "1ms", "5ms", "25ms", "125ms"],
    "Epsilon neighbors in time.",
)
flags.DEFINE_list(
    "resample_freqs",
    ["5T", "15T", "30T", "60T", "120T", "180T"],
    "Resample frequencies to compute OFIs over.",
)
flags.DEFINE_integer(
    "lookback",
    20,
    "Lookback period in days for rolling quantiles.",
)
# TODO: allow for (0.1, 0.9), (0.1, 0.5, 0.9) ...
flags.DEFINE_multi_float(
    "quantiles",
    [0.1, 0.9],
    "Quantiles for the features.",
    lower_bound=0.0,
    upper_bound=1.0,
)
ETF_TO_EQUITIES = {"SPY": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "BRK.B", "META", "TSLA", "UNH", "XOM", "JNJ", "JPM", "V", "PG", "LLY", "MA", "AVGO", "HD", "MRK", "CVX", "PEP", "ABBV", "KO", "COST", "PFE", "CRM", "MCD", "WMT", "TMO", "CSCO", "BAC", "AMD", "ACN", "ADBE", "ABT", "LIN", "CMCSA", "DIS", "ORCL", "NFLX", "WFC", "TXN", "DHR", "VZ", "NEE", "PM", "BMY", "RTX", "NKE", "HON", "UPS", "COP", "LOW", "UNP", "SPGI", "INTU", "AMGN", "QCOM", "IBM", "INTC", "SBUX", "BA", "PLD", "MDT", "GE", "AMAT", "GS", "CAT", "MS", "T", "NOW", "ELV", "ISRG", "MDLZ", "LMT", "BKNG", "BLK", "GILD", "DE", "SYK", "AXP", "TJX", "ADI", "ADP", "CVS", "MMC", "C", "VRTX", "AMT", "SCHW", "LRCX", "TMUS", "MO", "CB", "REGN", "ZTS", "MU", "SO", "PGR", "CI", "BSX", "FISV", "ETN", "BDX", "DUK", "PYPL", "SNPS", "EQIX", "CSX", "EOG", "TGT", "SLB", "AON", "CL", "CME", "HUM", "NOC", "ITW", "CDNS", "APD", "KLAC", "WM", "ICE", "ORLY", "CMG", "HCA", "ATVI", "MCK", "MMM", "SHW", "FDX", "EW", "GIS", "MPC", "PXD", "MCO", "CCI", "NSC", "FCX", "PNC", "ROP", "MSI", "KMB", "AZO", "MAR", "GD", "DG", "GM", "EMR", "SRE", "PSA", "F", "PSX", "NXPI", "EL", "DXCM", "APH", "MNST", "VLO", "FTNT", "AJG", "BIIB", "OXY", "ADSK", "USB", "AEP", "D", "PH", "JCI", "MRNA", "ECL", "TDG", "MCHP", "TFC", "ADM", "TRV", "CTAS", "AIG", "EXC", "CTVA", "ANET", "TT", "HSY", "COF", "IDXX", "TEL", "STZ", "CPRT", "HLT", "MSCI", "PCAR", "IQV", "O", "YUM", "AFL", "HES", "SYY", "DOW", "ON", "A", "WMB", "ROST", "XEL", "CNC", "WELL", "MET", "PAYX", "CARR", "VRSK", "NUE", "OTIS", "CHTR", "LHX", "AME", "DHI", "SPG", "ED", "EA", "NEM", "AMP", "KMI", "KR", "RMD", "CTSH", "FIS", "CSGP", "ROK", "DVN", "PPG", "FAST", "DD", "ILMN", "VICI", "KHC", "PEG", "CMI", "GWW", "BK", "PRU", "ALL", "MTD", "RSG", "GEHC", "DLTR", "KEYS", "ODFL", "BKR", "LEN", "ABC", "AWK", "HAL", "WEC", "CEG", "ZBH", "ACGL", "HPQ", "ANSS", "KDP", "DFS", "IT", "PCG", "DLR", "GPN", "VMC", "OKE", "EFX", "WST", "EIX", "ULTA", "MLM", "PWR", "ES", "WBD", "APTV", "FANG", "ALB", "SBAC", "AVB", "CBRE", "STT", "EBAY", "GLW", "URI", "TSCO", "XYL", "WTW", "TROW", "IR", "CDW", "FTV", "DAL", "CHD", "GPC", "ENPH", "LYB", "MPWR", "MKC", "CAH", "HIG", "TTWO", "WBA", "WY", "AEE", "BAX", "DTE", "VRSN", "MTB", "ALGN", "EQR", "FE", "STE", "IFF", "FSLR", "ETR", "CTRA", "DRI", "HOLX", "CLX", "EXR", "FICO", "PODD", "PPL", "INVH", "DOV", "HPE", "LH", "TDY", "COO", "LVS", "EXPD", "OMC", "NDAQ", "RJF", "CNP", "ARE", "BR", "K", "LUV", "FITB", "FLT", "VTR", "NVR", "RCL", "WAB", "MAA", "BALL", "CMS", "SEDG", "CAG", "ATO", "RF", "TYL", "GRMN", "HWM", "SWKS", "MOH", "SJM", "STLD", "IRM", "TRGP", "CINF", "LW", "UAL", "WAT", "PFG", "TER", "IEX", "PHM", "NTRS", "NTAP", "HBAN", "BRO", "MRO", "TSN", "FDS", "DGX", "RVTY", "AMCR", "EPAM", "IPG", "J", "EXPE", "JBHT", "RE", "CBOE", "AKAM", "BG", "BBY", "PTC", "LKQ", "SNA", "PAYC", "AVY", "ZBRA", "AES", "EQT", "ESS", "EVRG", "TXT", "CFG", "SYF", "AXON", "FMC", "TECH", "LNT", "POOL", "MGM", "CF", "WDC", "HST", "PKG", "UDR", "CHRW", "STX", "NDSN", "INCY", "MOS", "LYV", "TRMB", "KMX", "SWK", "WRB", "TAP", "CPT", "MAS", "BWA", "L", "CCL", "BF.B", "IP", "HRL", "VTRS", "TFX", "KIM", "NI", "DPZ", "APA", "ETSY", "JKHY", "LDOS", "WYNN", "PEAK", "CE", "CPB", "MKTX", "HSIC", "CRL", "TPR", "EMN", "GEN", "JNPR", "GL", "QRVO", "MTCH", "CDAY", "AAL", "PNR", "ALLE", "KEY", "FOXA", "ROL", "CZR", "FFIV", "PNW", "REG", "AOS", "BBWI", "UHS", "XRAY", "BIO", "HII", "NRG", "HAS", "RHI", "GNRC", "WHR", "NWSA", "PARA", "WRK", "BEN", "AAP", "BXP", "IVZ", "CTLT", "AIZ", "FRT", "NCLH", "SEE", "VFC", "ALK", "DXC", "DVA", "CMA", "OGN", "MHK", "RL", "ZION", "FOX", "LNC", "NWL", "NWS", "DISH", "VNT"], "XLF": ["BRK.B", "JPM", "V", "MA", "BAC", "WFC", "SPGI", "GS", "MS", "BLK", "AXP", "MMC", "C", "SCHW", "CB", "PGR", "FISV", "PYPL", "AON", "CME", "ICE", "MCO", "PNC", "AJG", "USB", "TFC", "TRV", "AIG", "COF", "MSCI", "AFL", "MET", "AMP", "FIS", "BK", "PRU", "ALL", "ACGL", "DFS", "GPN", "STT", "WTW", "TROW", "HIG", "MTB", "NDAQ", "RJF", "FLT", "FITB", "RF", "CINF", "PFG", "RE", "HBAN", "NTRS", "BRO", "FDS", "CBOE", "CFG", "SYF", "WRB", "L", "JKHY", "MKTX", "GL", "KEY", "BEN", "IVZ", "AIZ", "CMA", "ZION", "LNC"], "XLB": ["LIN", "APD", "SHW", "FCX", "ECL", "CTVA", "DOW", "NUE", "NEM", "PPG", "DD", "VMC", "MLM", "ALB", "LYB", "IFF", "BALL", "STLD", "AMCR", "AVY", "FMC", "CF", "PKG", "MOS", "IP", "CE", "EMN", "WRK", "SEE"], "XLK": ["MSFT", "AAPL", "NVDA", "AVGO", "CRM", "CSCO", "AMD", "ACN", "ADBE", "ORCL", "TXN", "INTU", "QCOM", "IBM", "INTC", "AMAT", "NOW", "ADI", "LRCX", "MU", "SNPS", "CDNS", "KLAC", "ROP", "MSI", "NXPI", "APH", "FTNT", "ADSK", "MCHP", "ANET", "TEL", "ON", "CTSH", "KEYS", "IT", "HPQ", "ANSS", "GLW", "CDW", "ENPH", "MPWR", "VRSN", "FSLR", "FICO", "HPE", "TDY", "SEDG", "TYL", "SWKS", "TER", "NTAP", "EPAM", "AKAM", "PTC", "ZBRA", "WDC", "STX", "TRMB", "JNPR", "GEN", "QRVO", "FFIV", "DXC"], "XLV": ["UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR", "BMY", "AMGN", "MDT", "ELV", "ISRG", "GILD", "SYK", "CVS", "VRTX", "REGN", "ZTS", "BSX", "CI", "BDX", "HUM", "HCA", "MCK", "EW", "DXCM", "BIIB", "MRNA", "IDXX", "IQV", "A", "CNC", "RMD", "ILMN", "MTD", "GEHC", "ABC", "ZBH", "WST", "CAH", "BAX", "ALGN", "STE", "HOLX", "PODD", "LH", "COO", "MOH", "WAT", "DGX", "RVTY", "TECH", "INCY", "TFX", "VTRS", "HSIC", "CRL", "UHS", "BIO", "XRAY", "CTLT", "DVA", "OGN"], "XLI": ["RTX", "HON", "UPS", "UNP", "BA", "GE", "CAT", "LMT", "DE", "ADP", "ETN", "CSX", "NOC", "ITW", "WM", "MMM", "FDX", "NSC", "GD", "EMR", "PH", "JCI", "TDG", "CTAS", "TT", "CPRT", "PCAR", "PAYX", "CARR", "VRSK", "OTIS", "LHX", "AME", "CSGP", "ROK", "FAST", "CMI", "GWW", "RSG", "ODFL", "EFX", "PWR", "URI", "XYL", "IR", "FTV", "DAL", "DOV", "EXPD", "BR", "LUV", "WAB", "HWM", "UAL", "IEX", "J", "JBHT", "SNA", "PAYC", "AXON", "TXT", "CHRW", "NDSN", "SWK", "MAS", "LDOS", "CDAY", "PNR", "AAL", "ALLE", "ROL", "AOS", "HII", "GNRC", "RHI", "ALK", "GEHC"], "XLU": ["NEE", "SO", "DUK", "SRE", "AEP", "D", "EXC", "XEL", "ED", "PEG", "AWK", "WEC", "CEG", "PCG", "EIX", "ES", "AEE", "DTE", "FE", "ETR", "PPL", "CNP", "CMS", "ATO", "AES", "EVRG", "LNT", "NI", "PNW", "NRG"], "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX", "ORLY", "CMG", "MAR", "AZO", "GM", "F", "HLT", "YUM", "ROST", "DHI", "LEN", "ULTA", "APTV", "EBAY", "TSCO", "GPC", "DRI", "LVS", "RCL", "NVR", "GRMN", "PHM", "EXPE", "BBY", "LKQ", "POOL", "MGM", "KMX", "CCL", "BWA", "ETSY", "DPZ", "WYNN", "TPR", "CZR", "BBWI", "HAS", "WHR", "AAP", "NCLH", "VFC", "MHK", "RL", "NWL"], "XLP": ["PG", "PEP", "KO", "COST", "MDLZ", "WMT", "PM", "MO", "TGT", "CL", "GIS", "KMB", "DG", "EL", "MNST", "ADM", "HSY", "STZ", "SYY", "KR", "KHC", "DLTR", "KDP", "CHD", "MKC", "WBA", "CLX", "K", "CAG", "SJM", "LW", "TSN", "BG", "TAP", "BF.B", "HRL", "CPB"], "XLE": ["XOM", "CVX", "EOG", "COP", "SLB", "MPC", "PXD", "PSX", "VLO", "OXY", "HES", "WMB", "KMI", "DVN", "BKR", "HAL", "OKE", "FANG", "CTRA", "TRGP", "MRO", "EQT", "APA"], "XLC": ["META", "GOOGL", "GOOG", "NFLX", "CMCSA", "ATVI", "TMUS", "VZ", "DIS", "CHTR", "EA", "T", "WBD", "TTWO", "OMC", "IPG", "LYV", "MTCH", "FOXA", "PARA", "NWSA", "FOX", "NWS", "DISH"], "IYR": ["PLD", "AMT", "EQIX", "CCI", "PSA", "O", "WELL", "SPG", "CSGP", "VICI", "DLR", "SBAC", "AVB", "CBRE", "WY", "EQR", "EXR", "INVH", "ARE", "VTR", "MAA", "SUI", "IRM", "WPC", "ESS", "UDR", "GLPI", "HST", "CPT", "KIM", "ELS", "LSI", "PEAK", "AMH", "REXR", "CUBE", "NLY", "REG", "LAMR", "COLD", "NNN", "Z", "EGP", "FR", "HR", "JLL", "BXP", "OHI", "FRT", "STAG", "BRX", "ADC", "STWD", "SRC", "AGNC", "AIRC", "MPW", "RYN", "RITM", "PCH", "BXMT", "NSA", "DOC", "CUZ", "KRC", "LXP", "HHC", "ZG", "OFC", "SBRA", "XTSLA", "EQC", "NHI", "VNO", "HIW", "DEI", "SLG", "JBGS", "OPEN", "MLPFT", "MARGIN_USD", "USD"]}  # fmt: skip

HYPERPARAMETER_ORDER = ["etf", "epsilon", "resample_freq", "quantiles", "lookback"]


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

FEATURE_NAMES_WITH_MARGINALS = (
    _FEATURE_NAMES
    + _FEATURE_FUNCTIONS
    + _MARGINAL_BEFORE_AFTER
    + _MARGINAL_SAME_SIGN_OPPOSITE_SIGN
)
