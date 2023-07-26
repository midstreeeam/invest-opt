import numpy as np

from optfolio import YEAR_BARS


def annualized_return(solutions: np.ndarray, returns_mean: np.ndarray) -> np.ndarray:
    """
    Calculate the annualized return for each solution in the population.

    Parameters:
    - solutions (np.ndarray): The population of portfolio allocations.
    - returns_mean (np.ndarray): The mean returns of the assets.

    Returns:
    - np.ndarray: The annualized returns for each solution.
    """
    returns = np.matmul(solutions, returns_mean)
    return (returns + 1) ** YEAR_BARS - 1

def annualized_volatility(solutions: np.ndarray, returns_cov: np.ndarray) -> np.ndarray:
    """
    Calculate the annualized volatility for each solution in the population.

    Parameters:
    - solutions (np.ndarray): The population of portfolio allocations.
    - returns_cov (np.ndarray): The covariance matrix of the asset returns.

    Returns:
    - np.ndarray: The annualized volatilities for each solution.
    """
    volatilities = np.sum(solutions * np.matmul(solutions, returns_cov), -1)
    return np.sqrt(volatilities * YEAR_BARS)

def unit_sum_constraint(solutions: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Ensure that the sum of allocations in each solution equals 1.

    Parameters:
    - solutions (np.ndarray): The population of portfolio allocations.
    - eps (float, optional): A small value to account for floating-point inaccuracies. Default is 1e-4.

    Returns:
    - np.ndarray: The amount by which each solution violates the unit sum constraint.
    """
    return np.clip(np.abs(np.sum(solutions, -1) - 1) - eps, 0, None)

def max_allocation_constraint(solutions: np.ndarray, max_allocation: float, eps: float = 1e-4):
    """
    Ensure that no asset's allocation in a solution exceeds the specified maximum allocation.

    Parameters:
    - solutions (np.ndarray): The population of portfolio allocations.
    - max_allocation (float): The maximum allowed allocation for any asset.
    - eps (float, optional): A small value to account for floating-point inaccuracies. Default is 1e-4.

    Returns:
    - np.ndarray: The amount by which each solution violates the max allocation constraint.
    """
    d = np.ones_like(solutions) * max_allocation
    return np.clip(np.sum(np.clip(solutions - d, 0, None), -1) - eps, 0, None)

