## Introduction
## Literature Review
## Methodology

To enhance the efficacy of the NSGA-II algorithm, three principal modifications are suggested. These recommendations emphasize (1) the normalization of population distribution, (2) the promotion of mutation and exploration as convergence approaches, and (3) the optimization of the initial population distribution. Additionally, throughout our implementation, we employ an [Improved Crowding Distance formula](https://arxiv.org/pdf/1811.12667.pdf) to further enhance performance.

We performed an empirical test using the data of 26 different stocks collected from various sectors for portfolio optimization, with the goal of pinpointing an optimal Pareto front.

#### Population Distribution

In the conventional NSGA-II algorithm, the **fronts** and the **crowding distance** are the two critical factors that determine whether a new individual will survive or an existing individual will become a parent after tournament selection.

While crowding distance and tournament selection are mechanisms intended to encourage population diversity, it is still possible for the population to cluster within a certain segment of the Pareto-optimal front, rather than achieving an equal distribution throughout. During tournament comparisons, the rank attributed to the non-domination front takes precedence over crowding distance. Consequently, if a subset of the population gravitates towards a specific region on the Pareto front that inherently has a superior rank, this subset is likely to dominate over other solutions with a lower rank in the non-dominance hierarchy. Such dominance has the potential to pull solutions of a lower rank towards its vicinity, culminating in a progressively denser concentration in these regions.

![[dist1.png]]

This diagram illustrates an optimal solution set established by the NSGA-II for the portfolio optimization task. The plot distinctly reveals that the solution is extremely dense in the center of the front, and excessively sparse on both sides. This uneven distribution is not the desired outcome. An imbalanced distribution of solutions implies that the algorithm might overlook potential viable solutions on both ends. On the other hand, solutions in the center might be over-explored (since this part may have already converged, an excess of computing resources can only lead to marginal improvements), which would result in a wastage of computational resources.

To address this, we refined the tournament selection strategy. In the traditional NSGA-II algorithm, candidates for the tournament were chosen randomly from the entire population set, which means the selection is normally distribute for the population rather than the possible solution space. If the population distribution is skewed, the tournament selection will be biased as well. Unlike [NSGA-III](https://ieeexplore.ieee.org/document/6600851) which use reference directions and introduce additional computational complexity, we don't "pin" the solutions around reference vector, but still offer them certain opportunity to explore.

The improved selection strategy is shown below
$$
\begin{align*} 
P(i) &= \frac{c(i)^b}{\sum_{j=1}^{n} c(j)^b} \quad \text{where } n \text{ is the number of solutions} \\ \\
T(i, j) &= \begin{cases} i & \text{if } f(i) < f(j) \text{ or } (f(i) = f(j) \text{ and } c(i) < c(j)) \\ 
j & \text{otherwise} \end{cases} \\ \\
R &= \{T(x_i, y_i) \mid (x_i, y_i) \text{ is the } i^{th} \text{ randomly selected pair based on } P \text{ for } i = 1, 2, ..., k \} \\ \\
\text{parents} &= (\max_{i \in R} f(i), \max_{j \in R} f(j)) 
\end{align*}
$$
Where
- $f$ is the array representing the fronts of each solution.
- $c$ is the array representing the crowding distances of each solution.
- $k$ is the number of candidates in the tournament.
- $b$ is the bias parameter, where 0 means no bias (then the algorithm will retrograde to normal NSGA-II), 1 means strictly pick solutions with highest crowding distance.

The novel strategy we propose is to bias the random selection in favor of less crowded solutions, making these more likely to be selected as tournament candidates. This approach will ensure that sparse areas (those needing more exploration) are more actively explored.

![[dist2.png]]

This plot displays the results produced by both the original NSGA-II and the NSGA-II with an optimized selection strategy on the same task . From the visualization, it's evident that the optimized version yields a more continuous and superior Pareto front.

#### Dynamic mutation parameters

In Evolutionary Algorithms, two critical parameters pertain to mutation: the mutation probability and sigma. The former dictates the likelihood of a mutation occurring, while the latter determines the magnitude of the mutation. Usually, as the generations increase, a decay mechanism will be introduced to reduce the mutation probability. This idea bears some resemblance to the optimizer and learning rate concepts in Artificial Neural Networks (ANN).

In our methodology, we dynamically adjust both the mutation probability and sigma throughout the evolutionary process to promote exploration and maintain diversity within the population. The foundational principle for this dynamic adaptation is straightforward: the denser the Pareto front becomes, the higher the likelihood and magnitude of mutations. By making the population more "energetic" or active to try to jump out of the local optimal as convergence nears, the algorithm may require more generations to reach convergence. However, this can increase the chances of discovering an optimal solution.

We've retained the decay mechanism, ensuring that over time the dynamic mutation probability and sigma gradually diminish, thereby not hindering the eventual convergence of solutions.

#### Population Initialization

In NSGA-II, the population is initialized randomly. For portfolio optimization tasks, the starting position of the population plays a crucial role. If random initialization serendipitously yields several promising solutions, the Pareto fronts can potentially improve. This is because these superior solutions will rapidly produce numerous offsprings, minimizing the time spent on exploration and increasing the likelihood of convergence to an optimal solution.

The size of the population is pivotal in this context. A larger population ensures more extensive coverage of the solution space, thereby increasing the probability of an advantageous start. However, a significant drawback is that larger populations demand considerable computational resources, which often isn't feasible.

A strategic approach to harness the benefits of both a good start from a larger population and the computational efficiency of a smaller one involves an initial "burst" of exploration. Start with an exceptionally large population size for just one generation to scout for optimal starting positions, and then scale down to a standard-sized population for subsequent exploration. This technique can markedly enhance result quality with negligible additional computational overhead.

Additionally, employing a combination of rank in the non-dominant front and crowding distance to downsize the population post-initialization yields dual benefits. Apart from achieving a superior Pareto front, the overall distribution of the population aligns more closely with the Pareto front. This proximity facilitates accelerated exploration in subsequent generations, optimizing the algorithm's efficiency and efficacy.

|Standard Initialization|Optimized Initialization|
| ----------- | ----------- |
| ![[init1.png]]| ![[init2.png]]       |

The two graphs vividly highlight the stark contrast between standard initialization and optimized initialization. It's clear that the optimized approach outperforms the standard one significantly. The blue dots symbolize random solutions, providing a visual of the solution space; red dots represent the optimal solutions identified, while the yellow dots signify other members of the population. Even though both scenarios encompass 3,000 individuals, the latter not only produces superior Pareto fronts but also boasts a more favorable distribution of the population.

## Results

For a portfolio optimization task, three different setups were executed and compared using the same dataset:

1. Standard NSGA-II
2. NSGA-II with selection optimization
3. NSGA-II enhanced with selection optimization, dynamic mutation, and optimized initialization.

Upon analyzing the Pareto fronts generated by each of these algorithms, we proceeded to select the solution with the highest Sharpe Ratio from each set. These chosen solutions were then subjected to a Monte Carlo Markov Chain (MCMC) analysis to further evaluate their predictive capabilities.

#### Optimized Solution

During the iteration, the hypervolume for perato front was used to track and visualize the performance of the three NSGA algorithms.

![[comp_hv.png]]

The graph clearly illustrates the superior performance of the fully optimized strategy, followed by the selection-optimized NSGA-II, while the standard NSGA-II trails behind.

From the visual representation:

1. **Initial Phase**: Both the standard NSGA-II and the selection-optimized variant start with nearly identical hypervolume values, suggesting a similar initial performance. In stark contrast, the fully optimized strategy — with its tailored initialization — begins its journey with a significantly enhanced hypervolume, indicating a promising outset.
    
2. **Evolutionary Progression**: The standard NSGA-II showcases two prominent increase in its hypervolume, signifying two key moments when its Pareto front witnessed substantial advancements. On the other hand, the two optimized versions depict more consistent and frequent upticks in their hypervolumes. This regularity in growth accentuates the effectiveness of the refined tournament selection strategy, affirming its role in directing the population's exploration of the solution space more proficiently.
    
3. **Convergence and Duration**: The algorithm equipped with dynamic mutation parameters exhibits a decelerated convergence rate, leading to an extended training duration. However, its hypervolume continues to expand, especially as it nears convergence. This persistence in growth underscores the advantage of dynamic mutation, allowing the algorithm to continually refine and potentially discover superior solutions even in the latter stages of the optimization process.

![[comp_solution.png]]

This graph shows the final solution sets generated by all three algorithms.