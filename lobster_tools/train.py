from absl import app
from absl import flags
from absl import logging

from lobster_tools import sim

FLAGS = flags.FLAGS

NPTS = flags.DEFINE_integer(
    name='npts', default=1000, help='Number of data points')

def main(_):
    ans = sim.get_numpy_array(FLAGS.npts)
    logging.info(ans[-1])
    logging.info(FLAGS.run_simulation)

if __name__ == "__main__":
    app.run(main)