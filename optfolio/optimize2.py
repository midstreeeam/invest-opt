import time
from typing import Tuple
import math

import numpy as np

from optfolio.objectives import *
from optfolio.n_nsga2 import *

class Optimizer:

    def __init__(
            self, 
            population_size: int = 5000, 
            max_iter: int = 100, 
            mutation_p: float = 0.3, 
            mutation_p_decay: float = 0.98, 
            mutation_sigma: float = 0.1, 
            verbose: bool = False,
            opt: bool = True,
            init_boost_scaler: int = 12
    ):
        self._population_size = population_size
        self._max_iter = max_iter
        self._mutation_p = mutation_p
        self._mutation_p_decay = mutation_p_decay
        self._mutation_sigma = mutation_sigma
        self._verbose = verbose
        self._dynamic_sigma = opt
        self._dynamic_p = opt
        self._init_boost = opt
        self._boost_scaler = init_boost_scaler
        self.filename = f"{population_size}_opt"
        if opt:
            self.filename = f"{population_size}_opt2"
        print("debug")


    def run(self, returns: np.ndarray, max_allocation: float = None) -> Tuple[np.ndarray, dict]:
        stats = {
            'return': {'min': [], 'max': [], 'avg': []},
            'volatility': {'min': [], 'max': [], 'avg': []},
            'constraints_violation': {'min': [], 'max': [], 'avg': []},
            'hv': [],
            'time_per_generation': []
        }
        returns_mean = np.mean(returns, 0)
        returns_cov = np.cov(returns.T)
        population = self._init_population(len(returns_mean))
        
        # Calculate objectives for the initial population
        return_obj = annualized_return(population, returns_mean)
        volatility_obj = annualized_volatility(population, returns_cov)
        constraints_val = unit_sum_constraint(population)
        if max_allocation is not None:
            constraints_val += max_allocation_constraint(population, max_allocation)
        fronts, crowding_distances = non_dominated_fronts(return_obj, volatility_obj, constraints_val)

        # Start the optimization loop
        for gen_idx in range(self._max_iter):
            gen_start_time = time.time()
            mutation_p = self._mutation_p * (self._mutation_p_decay ** gen_idx)
            # print(mutation_p)
            offspring = np.empty_like(population)
            for i in range(self._population_size):
                (p1_idx, p2_idx) = tournament_selection(fronts, crowding_distances)
                offspring[i, :] = flat_crossover(population[p1_idx], population[p2_idx])
                if np.random.uniform() < mutation_p:
                    offspring[i, :] = gaussian_mutation(offspring[i, :], sigma=self._mutation_sigma)
            offspring = np.clip(offspring, 0, 1)
            population = np.concatenate((population, offspring), axis=0)
            return_obj = annualized_return(population, returns_mean)
            volatility_obj = annualized_volatility(population, returns_cov)
            constraints_val = unit_sum_constraint(population)
            if max_allocation is not None:
                constraints_val += max_allocation_constraint(population, max_allocation)
            
            fronts, crowding_distances = non_dominated_fronts(return_obj, volatility_obj, constraints_val)
            population, fronts, crowding_distances, return_obj, volatility_obj, constraints_val = select_top_individuals(
                population, fronts, crowding_distances, return_obj, volatility_obj, constraints_val)

            pareto_front_data = np.column_stack((-volatility_obj[fronts == 0],return_obj[fronts == 0]))
            hv = hypervolume(pareto_front_data)

            # Update statistics
            self._append_stats(stats, return_obj[fronts == 0], volatility_obj[fronts == 0],
                               constraints_val[fronts == 0], hv, float(time.time() - gen_start_time))
                      
            if self._verbose:
                print(
                    f"===============\n"
                    f"Generation: {gen_idx},\n"
                    f"Time: {stats['time_per_generation'][-1]:.2f}, \n"
                    f"N pareto solutions: {np.sum(fronts == 0)}, \n"
                    f"Hypervolume: {hv}\n"
                )

        pareto_front_ids = np.argwhere(fronts == 0).reshape((-1,))
        return population[pareto_front_ids], stats

    def _init_population(self, n_assets: int) -> np.ndarray:
        if self._init_boost:
            population = np.random.uniform(0, 1, size=(self._population_size*self._boost_scaler, n_assets))
        else:
            population = np.random.uniform(0, 1, size=(self._population_size, n_assets))

        return population / np.sum(population, 1).reshape((-1, 1))

    def _append_stats(
        self, 
        stats: dict, 
        return_obj: np.ndarray, 
        volatility_obj: np.ndarray, 
        constraints_val: np.ndarray, 
        hv: float, 
        tpg: float
    ):
        stats['hv'].append(hv)
        stats['time_per_generation'].append(tpg)
        for (k, v) in [('return', return_obj), ('volatility', volatility_obj), ('constraints_violation', constraints_val)]:
            stats[k]['min'].append(np.min(v))
            stats[k]['max'].append(np.max(v))
            stats[k]['avg'].append(np.mean(v))



    def run_generator(self, returns: np.ndarray, max_allocation: float = None) -> Tuple[np.ndarray, dict]:
        stats = {
            'return': {'min': [], 'max': [], 'avg': []},
            'volatility': {'min': [], 'max': [], 'avg': []},
            'constraints_violation': {'min': [], 'max': [], 'avg': []},
            'hv': [],
            'time_per_generation': []
        }
        returns_mean = np.mean(returns, 0)
        returns_cov = np.cov(returns.T)

        # if opt, the init population will be boosted
        population = self._init_population(len(returns_mean))
        
        # Calculate objectives for the initial population
        return_obj = annualized_return(population, returns_mean)
        volatility_obj = annualized_volatility(population, returns_cov)
        constraints_val = unit_sum_constraint(population)
        if max_allocation is not None:
            constraints_val += max_allocation_constraint(population, max_allocation)
        fronts, crowding_distances = non_dominated_fronts(return_obj, volatility_obj, constraints_val)

        print(len(population))
        print(len(fronts[fronts==0]))

        # drop population to normal if it is boosted
        if self._init_boost:
            population, fronts, crowding_distances, return_obj, volatility_obj, constraints_val = select_top_individuals(
                population, fronts, crowding_distances, return_obj, volatility_obj, constraints_val, target_scaler=self._boost_scaler)
        
        print(len(population))
        print(len(fronts[fronts==0]))

        # Start the optimization loop
        for gen_idx in range(self._max_iter):
            gen_start_time = time.time()

            # get p
            mutation_p = self._mutation_p
            if self._dynamic_p:
                mutation_p = adjust_mutation_p(fronts,crowding_distances)
            mutation_p *= (self._mutation_p_decay ** gen_idx)

            # get sigma
            if self._dynamic_sigma:
                self._mutation_sigma = adjust_mutation_sigma(fronts,crowding_distances)

            # print(mutation_p)
            offspring = np.empty_like(population)
            for i in range(self._population_size):
                (p1_idx, p2_idx) = tournament_selection(fronts, crowding_distances)
                offspring[i, :] = flat_crossover(population[p1_idx], population[p2_idx])
                if np.random.uniform() < mutation_p:
                    offspring[i, :] = gaussian_mutation(offspring[i, :], sigma=self._mutation_sigma)
            offspring = np.clip(offspring, 0, 1)
            population = np.concatenate((population, offspring), axis=0)
            return_obj = annualized_return(population, returns_mean)
            volatility_obj = annualized_volatility(population, returns_cov)
            constraints_val = unit_sum_constraint(population)
            if max_allocation is not None:
                constraints_val += max_allocation_constraint(population, max_allocation)
            
            fronts, crowding_distances = non_dominated_fronts(return_obj, volatility_obj, constraints_val)
            population, fronts, crowding_distances, return_obj, volatility_obj, constraints_val = select_top_individuals(
                population, fronts, crowding_distances, return_obj, volatility_obj, constraints_val)

            pareto_front_data = np.column_stack((-volatility_obj[fronts == 0],return_obj[fronts == 0]))
            hv = hypervolume(pareto_front_data)

            # Update statistics
            self._append_stats(stats, return_obj[fronts == 0], volatility_obj[fronts == 0],
                               constraints_val[fronts == 0], hv, float(time.time() - gen_start_time))
                      
            if self._verbose:
                print(
                    f"===============\n"
                    f"Generation: {gen_idx},\n"
                    f"Time: {stats['time_per_generation'][-1]:.2f}, \n"
                    f"N pareto solutions: {np.sum(fronts == 0)}, \n"
                    f"Hypervolume: {hv}, \n"
                    f"Mutation P: {mutation_p}, \n"
                    f"Mutation Sigma: {self._mutation_sigma}"
                )
                
            # Yield the current Pareto front solutions
            pareto_front_ids = np.argwhere(fronts == 0).reshape((-1,))
            other_ids = np.argwhere(fronts != 0).reshape((-1,))
            yield population[pareto_front_ids], population[other_ids], stats

            # early stop
            if len(fronts[fronts==0]) >= len(population)/4:
                break

        pareto_front_ids = np.argwhere(fronts == 0).reshape((-1,))
        return population[pareto_front_ids], population[other_ids], stats
