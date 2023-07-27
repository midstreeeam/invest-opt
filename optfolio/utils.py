import numba as nb
import numpy as np

from optfolio import HV_REFERENCE

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

def hypervolume(pareto_front, reference_point = HV_REFERENCE):
    """
    Compute the hypervolume indicator for a 2D Pareto front.
    
    Parameters:
    - pareto_front: A numpy array of shape (n, 2) where n is the number of points in the Pareto front.
    - reference_point: A 2D point used as a reference. The hypervolume is computed with respect to this point.
    
    Returns:
    - The hypervolume indicator.
    """
    # Sort the Pareto front by the first objective
    sorted_front = pareto_front[np.argsort(pareto_front[:, 0])]
    # print(sorted_front)

    # Initialize hypervolume to zero
    hv = 0.0
    
    # For each point in the sorted Pareto front
    for i in range(len(sorted_front)):
        if i == 0:
            width = reference_point[0] - sorted_front[i][0]
        else:
            width = sorted_front[i - 1][0] - sorted_front[i][0]
        height = reference_point[1] - sorted_front[i][1]
        hv += width * height
    return hv

def adjust_mutation_p(fronts, crowding_distances, min_p: float = 0.01, max_p: float = 0.5):
    """Dynamically adjust the mutation sigma based on the average crowding distance."""
    fronts_distances = crowding_distances[fronts == 0]
    distances = fronts_distances[fronts_distances != np.inf]
    avg_crowding_distance = np.mean(distances)
    range_ = distances.max() - distances.min()
    normalized_crowding_distance = avg_crowding_distance - distances.min()
    if range_ != 0:
        normalized_crowding_distance = (avg_crowding_distance - distances.min()) / range_
    # print(normalized_crowding_distance)
    p = max_p - (max_p - min_p) * normalized_crowding_distance
    return p

def adjust_mutation_sigma(fronts, crowding_distances, min_sigma: float = 0.1, max_sigma: float = 2.0):
    """Dynamically adjust the mutation sigma based on the average crowding distance."""
    fronts_distances = crowding_distances[fronts == 0]
    distances = fronts_distances[fronts_distances != np.inf]
    avg_crowding_distance = np.mean(distances)
    range_ = distances.max() - distances.min()
    normalized_crowding_distance = avg_crowding_distance - distances.min()
    if range_ != 0:
        normalized_crowding_distance = (avg_crowding_distance - distances.min()) / range_
    # print(normalized_crowding_distance)
    sigma = max_sigma - (max_sigma - min_sigma) * normalized_crowding_distance
    return sigma