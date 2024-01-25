from absl import logging

import numpy as np

def get_numpy_array(npts):
    logging.info(npts)
    return np.arange(npts)