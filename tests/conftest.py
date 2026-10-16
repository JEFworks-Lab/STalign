"""pytest configuration shared by the test suite.

Imports STalign from this source tree, draws figures off screen and limits the number of
threads (the suite must stay light on shared machines).
"""
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')

import matplotlib  # noqa: E402

matplotlib.use('Agg')

import pytest  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

torch.set_num_threads(min(4, torch.get_num_threads()))


@pytest.fixture(autouse=True)
def _close_figures():
    """Close the figures a test created."""
    yield
    import matplotlib.pyplot as plt
    plt.close('all')
