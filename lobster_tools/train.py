from absl import app
from absl import flags
from absl import logging

from lobster_tools import sim

NPTS = flags.DEFINE_integer(
    name='npts', default=1000, help='Number of data points')

def main(_):
    ans = sim.get_numpy_array(NPTS.value)
    logging.info(ans[-1])


if __name__ == "__main__":
    app.run(main)