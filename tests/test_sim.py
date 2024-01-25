from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from lobster_tools import geo

class GeoTest(parameterized.TestCase):

  @parameterized.parameters(
      {'npts': 100},
      {'npts': 1_000})
  def test_get_numpy_array(self, npts):
    ans = geo.get_numpy_array(npts)
    self.assertIsInstance(ans, np.ndarray)

  def test_return_true(self):
    self.assertTrue(True)

if __name__ == '__main__':
  absltest.main()