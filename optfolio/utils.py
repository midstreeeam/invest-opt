import numba as nb
import numpy as np

@nb.jit(nopython=True)
def is_close_to_zero(x):
    return x <= 1e-8

@nb.jit(nopython=True)
def select_index_by_weight(cum_weights: np.ndarray) -> int:
    """Selects an index based on cumulative weights."""
    rand_val = np.random.rand()
    for i in range(len(cum_weights)):
        if rand_val < cum_weights[i]:
            return i
    return len(cum_weights) - 1