from absl.testing import absltest
from absl.testing import parameterized

class AnotherTest(absltest.TestCase):

  def test_return_true(self):
    self.assertTrue(True)

if __name__ == '__main__':
  absltest.main()