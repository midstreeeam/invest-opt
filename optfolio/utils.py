import numba as nb
import numpy as np

@nb.jit(nopython=True)
def is_close_to_zero(x):
    return x <= 1e-8

@nb.jit(nopython=True)
def weighted_random_choice(probabilities: np.ndarray) -> int:
    """
    Choose an index based on the given probabilities.
    
    Parameters:
    - probabilities (np.ndarray): Array of probabilities for each index.
    
    Returns:
    - int: Chosen index.
    """
    cumulative_probs = np.cumsum(probabilities)
    random_value = np.random.rand()
    for idx, prob in enumerate(cumulative_probs):
        if random_value < prob:
            return idx
    return len(probabilities) - 1  # Return the last index if none found