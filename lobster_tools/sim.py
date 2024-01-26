from absl import flags
from absl import logging

import numpy as np

RUN_SIMULATION = flags.DEFINE_bool('run_simulation', True, 'Run simulation or not.')

def get_numpy_array(npts):
    logging.info(npts)
    return np.arange(npts)