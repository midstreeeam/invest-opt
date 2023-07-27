from typing import Tuple
import numpy as np
import numba as nb

from optfolio.utils import *


@nb.jit(nopython=True)
def non_dominated_fronts(return_obj: np.ndarray, volatility_obj: np.ndarray, constraints_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the non-dominated fronts and crowding distances for a given population.

    Parameters:
    - return_obj (np.ndarray): Array of annualized returns for each solution.
    - volatility_obj (np.ndarray): Array of annualized volatilities for each solution.
    - constraints_val (np.ndarray): Array of constraint violation values for each solution.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Arrays representing the front assignments and crowding distances for each solution.
    """

    domination_counts = np.zeros(len(return_obj), dtype=np.int32)
    domination_ids = []

    for i in range(len(return_obj)):
        dominates_ids = np.asarray([
            j for j in range(len(return_obj))
            if (
                # Both feasible
                is_close_to_zero(constraints_val[i]) and is_close_to_zero(constraints_val[j])) and \
            ((return_obj[i] >= return_obj[j]) and (volatility_obj[i] <= volatility_obj[j]) and ((return_obj[i] > return_obj[j]) or (volatility_obj[i] < volatility_obj[j]))
             ) or (
                # Both infeasible
                (not is_close_to_zero(constraints_val[i] and not is_close_to_zero(constraints_val[j]))) and \
                (constraints_val[i] < constraints_val[j])
            ) or (
                # ith feasible and jth not
                is_close_to_zero(constraints_val[i]) and not is_close_to_zero(
                    constraints_val[j])
            )
        ], dtype=np.uint32)

        domination_ids.append(dominates_ids)
        domination_counts[dominates_ids] += 1

    fronts = np.empty(len(return_obj), dtype=np.uint32)
    crowding_distances = np.empty(len(return_obj), dtype=np.float32)
    front_id = 0
    while np.any(domination_counts >= 0):
        front_ids = np.asarray([i for i, c in enumerate(
            domination_counts) if c == 0], dtype=np.uint32)
        fronts[front_ids] = front_id
        crowding_distances[front_ids] = _crowding_distance(
            return_obj[front_ids], volatility_obj[front_ids])
        domination_counts[front_ids] -= 1
        for idx in front_ids:
            domination_counts[domination_ids[idx]] -= 1
        front_id += 1

    return fronts, crowding_distances


@nb.jit(nopython=True)
def _crowding_distance(return_obj: np.ndarray, volatility_obj: np.ndarray) -> np.ndarray:
    """
    Compute the crowding distance for a set of solutions based on their objectives.

    Parameters:
    - return_obj (np.ndarray): Array of annualized returns for each solution.
    - volatility_obj (np.ndarray): Array of annualized volatilities for each solution.

    Returns:
    - np.ndarray: Array of crowding distances for each solution.
    """

    # Using Improved Crowding Distance formula (2) from https://arxiv.org/pdf/1811.12667.pdf
    distances = np.zeros(len(return_obj), dtype=np.float32)
    for objective_values, mul in [(return_obj, -1), (volatility_obj, 1)]:
        sort_ids = np.argsort(mul * objective_values)
        sorted_values = objective_values[sort_ids]
        distances[sort_ids[:-1]] += (sorted_values[1:] - sorted_values[:-1]) / \
            (sorted_values[-1] - sorted_values[0] + 1e-8)
        distances[sort_ids[-1]] = np.inf

    return distances


@nb.jit(nopython=True)
def _tournament_is_better(a_front: int, a_crowd: int, b_front: int, b_crowd: int) -> bool:
    """
    Determine if one solution is better than another based on their front and crowding distance.

    Parameters:
    - a_front (int): Front assignment of the first solution.
    - a_crowd (int): Crowding distance of the first solution.
    - b_front (int): Front assignment of the second solution.
    - b_crowd (int): Crowding distance of the second solution.

    Returns:
    - bool: True if the first solution is better, False otherwise.
    """

    return (a_front < b_front) or ((a_front == b_front) and (a_crowd < b_crowd))

# @nb.jit(nopython=True)
def tournament_selection(fronts: np.ndarray, crowding_distances: np.ndarray, k: int = 50, b: float = 0.5) -> Tuple[int, int]:
    """
    Perform tournament selection to choose two parent solutions for crossover.

    Parameters:
    - fronts (np.ndarray): Array of front assignments for each solution.
    - crowding_distances (np.ndarray): Array of crowding distances for each solution.
    - k (int, optional): Number of candidates to consider in the tournament. Default is 50.
    - b (float, optional): Bias towards selecting candidates with larger crowding distances. Ranges from 0.0 to 1.0.

    Returns:
    - Tuple[int, int]: Indices of the two selected parent solutions.
    """
    
    # Ensure all crowding distances are non-negative and not NaN
    crowding_distances = np.maximum(crowding_distances, 0)
    crowding_distances = np.nan_to_num(crowding_distances)
    
    # Bias the selection based on crowding distances
    biased_probs = crowding_distances ** b
    biased_probs = np.nan_to_num(biased_probs)
    
    total = np.sum(biased_probs)
    
    # Avoid division by zero and NaN
    if total == 0 or np.isnan(total):
        probabilities = np.ones_like(crowding_distances) / len(crowding_distances)
    else:
        probabilities = biased_probs / total
        probabilities = np.nan_to_num(probabilities)
    
    candidate_ids = np.random.choice(len(fronts), size=(k, 2), p=probabilities, replace=False)
    best_i, best_j = -1, -1
    for idx in range(k):
        i, j = candidate_ids[idx, :]
        if best_i < 0:
            best_i = i
            best_j = j
        else:
            if _tournament_is_better(fronts[i], crowding_distances[i], fronts[best_i], crowding_distances[best_i]):
                best_i = i
            if _tournament_is_better(fronts[j], crowding_distances[j], fronts[best_j], crowding_distances[best_j]):
                best_j = j

    return best_i, best_j


@nb.jit(nopython=True)
def select_top_individuals(
    population: np.ndarray,
    fronts: np.ndarray,
    crowding_distances: np.ndarray,
    return_obj: np.ndarray,
    volatility_obj: np.ndarray,
    constraints_val: np.ndarray,
    target_scaler: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Select the top individuals from a population based on non-domination and crowding distance.

    Parameters:
    - population (np.ndarray): The current population of solutions.
    - fronts (np.ndarray): Array of front assignments for each solution.
    - crowding_distances (np.ndarray): Array of crowding distances for each solution.
    - return_obj (np.ndarray): Array of annualized returns for each solution.
    - volatility_obj (np.ndarray): Array of annualized volatilities for each solution.
    - constraints_val (np.ndarray): Array of constraint violation values for each solution.

    Returns:
    - Tuple[np.ndarray, ...]: Arrays representing the new population, their front assignments, crowding distances, and objective values.
    """
    
    target_size = int(population.shape[0] / target_scaler)
    added_individuals_count = 0
    front_id = 0

    new_population = np.empty(
        (target_size, population.shape[1]), dtype=population.dtype)
    new_fronts = np.empty((target_size,), dtype=fronts.dtype)
    new_crowding_distances = np.empty(
        (target_size,), dtype=crowding_distances.dtype)
    new_return_obj = np.empty((target_size,), dtype=return_obj.dtype)
    new_volatility_obj = np.empty((target_size,), dtype=volatility_obj.dtype)
    new_constraints_val = np.empty((target_size,), dtype=constraints_val.dtype)

    while added_individuals_count < target_size:
        front_ids = np.asarray([i for i, f in enumerate(
            fronts) if f == front_id], dtype=np.uint32)
        front_id += 1
        if added_individuals_count + len(front_ids) > target_size:
            front_ids = front_ids[np.argsort(crowding_distances[front_ids])][:(
                target_size-added_individuals_count)]

        for idx in front_ids:
            new_population[added_individuals_count] = population[idx]
            new_fronts[added_individuals_count] = fronts[idx]
            new_crowding_distances[added_individuals_count] = crowding_distances[idx]
            new_return_obj[added_individuals_count] = return_obj[idx]
            new_volatility_obj[added_individuals_count] = volatility_obj[idx]
            new_constraints_val[added_individuals_count] = constraints_val[idx]
            added_individuals_count += 1

    return new_population, new_fronts, new_crowding_distances, new_return_obj, new_volatility_obj, new_constraints_val


@nb.jit(nopython=True)
def flat_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Perform flat crossover between two parent solutions.

    Parameters:
    - p1 (np.ndarray): The first parent solution.
    - p2 (np.ndarray): The second parent solution.

    Returns:
    - np.ndarray: The offspring solution.
    """
    
    g = np.stack((p1, p2))
    hc_lb, hc_ub = np.empty_like(p1), np.empty_like(p2)
    for i in range(len(p1)):
        hc_lb[i] = min(p1[i], p2[i])
        hc_ub[i] = max(p1[i], p2[i])

    rand = np.random.uniform(0, 1, size=p1.shape)
    return rand * (hc_ub - hc_lb) + hc_lb


@nb.jit(nopython=True)
def gaussian_mutation(x: np.ndarray, sigma: float = 1) -> np.ndarray:
    """
    Apply Gaussian mutation to a solution.

    Parameters:
    - x (np.ndarray): The original solution.
    - sigma (float, optional): The standard deviation of the Gaussian distribution used for mutation. Default is 1.

    Returns:
    - np.ndarray: The mutated solution.
    """

    return x + np.random.normal(0, sigma, size=x.shape)
