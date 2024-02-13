"""Undo the dictionary nesting that I did when saving the files"""

from pathlib import Path
import pickle
import shutil


_OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def main():
    prev_dir = _OUTPUT_DIR / "2024-02-11"

    etfs = [x.name for x in prev_dir.glob("*")]

    etfs.remove("XLK")  # not done running yet
    print(etfs)

    new_dir = _OUTPUT_DIR / "2024-02-12"
    new_dir.mkdir(exist_ok=True)

    for etf in etfs:
        print("****** ETF ******")
        print(etf)

        etf_dir = new_dir / etf
        etf_dir.mkdir()

        with open(
            prev_dir / etf / "tolerance_to_neighbors_and_features.pkl", "rb"
        ) as f:
            features_dict = pickle.load(f)

        for tol, dfs in features_dict.items():
            print(f"Tolerance: {tol}")
            tol_dir = etf_dir / tol
            tol_dir.mkdir()

            neighbors = dfs["neighbors"]
            features = dfs["features"]

            print("features.head()")
            print(features.head())

            neighbors.to_pickle(tol_dir / "neighbors.pkl")
            features.to_pickle(tol_dir / "features.pkl")


def just_copy_executions():
    prev_dir = _OUTPUT_DIR / "2024-02-11"
    etfs = [x.name for x in prev_dir.glob("*")]

    etfs.remove('XLK') # not done runnng yet
    etfs.remove('XLB') # just ran this already

    print(etfs)

    new_dir = _OUTPUT_DIR / "2024-02-12"

    for etf in etfs:
        print("****** ETF ******")
        print(etf)

        # copy etf_exeuctions.pkl
        shutil.copy(
            src=prev_dir / etf / "etf_executions.pkl",
            dst=new_dir / etf / "etf_executions.pkl",
        )
        shutil.copy(
            src=prev_dir / etf / "equity_executions.pkl",
            dst=new_dir / etf / "equity_executions.pkl",
        )
    print('done')

if __name__ == "__main__":
    # main()
    just_copy_executions()
