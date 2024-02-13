import os
import shutil
import tempfile

from absl.testing import absltest

from lobster_tools.preprocessing import FolderInfo, infer_ticker_date_ranges


class TestInferTickerDateRanges(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.dir = tempfile.mkdtemp()

        self.dummy_files = [
            "_data_dwn_32_302__AAPL_2021-01-01_2021-12-31_10.7z",
            "_data_dwn_32_302__MSFT_2021-01-01_2021-12-31_10.7z",
            "_data_dwn_32_302__SPY_2021-01-01_2021-12-31_10.7z",
        ]

        for file_name in self.dummy_files:
            file_path = os.path.join(self.dir, file_name)
            with open(file_path, "w") as _:
                pass

        self.result = infer_ticker_date_ranges(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir)
        super().tearDown()

    def test_length(self):
        self.assertLen(self.result, len(self.dummy_files))

    def test_ticker_names(self):
        self.assertSetEqual({"SPY", "AAPL", "MSFT"}, {x.ticker for x in self.result})

    def test_start_date(self):
        self.assertEqual(self.result[0].start_date, "2021-01-01")

    def test_type(self):
        self.assertIsInstance(self.result[0], FolderInfo)


if __name__ == "__main__":
    absltest.main()
