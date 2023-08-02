## Introduction

In the field of financial management, portfolio optimization stands as a cornerstone, striving to craft the ideal balance of assets that maximizes returns while minimizing risks. However, achieving this balance is often a complex process, with numerous variables and objectives to consider, where algorithms play an instrumental role. One such powerful algorithm, the NSGA-II[1], has been lauded for its capabilities in multi-objective optimization scenarios. Yet, as the complexities of financial markets evolve and the demands of investors become more nuanced, there's an urgent need to further refine and adapt the computational strategies, and there is room for NSGA-II's further enhancement. This article aims to bridge this gap by proposing and examining a set of principal modifications to the NSGA-II algorithm. Through a series of modifications, including improved tournament selection strategies, dynamic mutation parameters, and optimized population initialization, we hope to push the boundaries of what's possible in generating optimal asset allocations. In the subsequent sections, we introduce a literature review detailing portfolio optimization algorithms and Multi-objective Evolutionary Algorithms (MOEAs). The article then elucidates our proposed methodology and present the contrast between traditional and our enhanced NSGA-II approaches. We further employ the Monte Carlo Markov Chain (MCMC) for a ten-year return prediction comparison. In conclusion section, we offer a thorough evaluation and discussion of our findings.
## Literature Review

Portfolio optimization has consistently anchored financial management and investment analysis discussions over the past decades. Classical Linear Programming (LP) models, such as Markowitz's Mean-Variance Optimization (MVO) and the Black-Litterman Model, laid the foundation for understanding risk-return dynamics. As the financial landscape evolved, researchers began exploring advanced strategies like [machine learning approach](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2016.2644)[2] and [deep learning approach](https://www.pm-research.com/content/iijjfds/early/2020/08/26/jfds20201042.full)[3] to enhance portfolio performance.

However, as pointed out by [Renata Mansini et al.](https://www.sciencedirect.com/science/article/pii/S0377221713007194)[4], at the multiple criteria nature of the portfolio problem, a single LP model might not capture all the nuances of real-world portfolio optimization. And for models using Artifical Neuron Network (ANN), it's hard for them to generate the entire perato front base on NN's structural limitation.

This brings our attention to Multi-objective Evolutionary Algorithms (MOEAs), specifically [NSGA-II](https://chat.openai.com/c/c90f90a3-b283-4ff1-9489-768a1d08e1cf)[1]. Renowned for its capability to mirror real-world problems, NSGA-II presents an entire Pareto front instead of just a single optimum. Following its success, several augmentations emerged. [MNSGA-II](https://www.sciencedirect.com/science/article/pii/S0142061511000421)[5] introduced the Dynamic Crowding Distance (DCD) to maintain diversity in populations, while a [Novel Ranking Scheme](https://arxiv.org/abs/1002.4005) [6] optimized algorithmic time complexity.  [NSGA-III](https://ieeexplore.ieee.org/document/6600851)[7] and its variants, such as [U-NSGA-III](https://link.springer.com/chapter/10.1007/978-3-319-15892-1_3) [8] and [θ-NSGA-III](https://ieeexplore.ieee.org/abstract/document/8936438)[9], inherited NSGA-II's core principles but substituted crowding distance with reference directions to bolster diversity. Further enhancements came from [Zimin Gu and Gaige Wang](https://www.sciencedirect.com/science/article/pii/S0167739X19320552)[10], who integrated information feedback mechanisms with NSGA-III for large-scale multi-objective optimization.

While NSGA-III shows advancements over NSGA-II, the latter still garners considerable attention among academic researchers. This is partly because NSGA-III is only one possible evolution for traditional NSGA-II and there are many other possibilities. Recent studies have delved into the [acceleration of the crossover process](https://ojs.aaai.org/index.php/AAAI/article/view/26461)[11] within NSGA-II, while others have proved its [non-trivial lower bound](https://ojs.aaai.org/index.php/AAAI/article/view/26462)[12]. Notably, a [contemporary research piece](https://www.sciencedirect.com/science/article/pii/S0142061511000421)[13] combined deep learning with NSGA-II, addressing emission dispatch challenges.

In light of these developments, our intention is clear: to revisit the core NSGA-II algorithm and enhance its efficiency through strategic modifications. Unlike NSGA-III which "pin" the solutions to the solution space uniformly using strict rule, our interest lies in guiding individuals to explore the solution space more intelligently and effectively.
## Methodology

To enhance the efficacy of the NSGA-II algorithm, three principal modifications are suggested. These recommendations emphasize (1) the normalization of population distribution, (2) the promotion of mutation and exploration as convergence approaches, and (3) the optimization of the initial population distribution. Additionally, throughout our implementation, we employ an [Improved Crowding Distance formula](https://arxiv.org/abs/1811.12667)[13] to further enhance performance.

We performed an empirical test using the data of 26 different stocks collected from various sectors for portfolio optimization, with the goal of pinpointing an optimal Pareto front.
#### Population Distribution

In the conventional NSGA-II algorithm, the **fronts** and the **crowding distance** are the two critical factors that determine whether a new individual will survive or an existing individual will become a parent after tournament selection.

While crowding distance and tournament selection are mechanisms intended to encourage population diversity, it is still possible for the population to cluster within a certain segment of the Pareto-optimal front, rather than achieving an equal distribution throughout. During tournament comparisons, the rank attributed to the non-domination front takes precedence over crowding distance. Consequently, if a subset of the population gravitates towards a specific region on the Pareto front that inherently has a superior rank, this subset is likely to dominate over other solutions with a lower rank in the non-dominance hierarchy. Such dominance has the potential to pull solutions of a lower rank towards its vicinity, culminating in a progressively denser concentration in these regions.

![[dist1.png]]

This diagram illustrates an optimal solution set established by the NSGA-II for the portfolio optimization task. The plot distinctly reveals that the solution is extremely dense in the center of the front, and excessively sparse on both sides. This uneven distribution is not the desired outcome. An imbalanced distribution of solutions implies that the algorithm might overlook potential viable solutions on both ends. On the other hand, solutions in the center might be over-explored (since this part may have already converged, an excess of computing resources can only lead to marginal improvements), which would result in a wastage of computational resources.

To address this, we refined the tournament selection strategy. In the traditional NSGA-II algorithm, candidates for the tournament were chosen randomly from the entire population set, which means the selection is normally distribute for the population rather than the possible solution space. If the population distribution is skewed, the tournament selection will be biased as well. Unlike [NSGA-III](https://ieeexplore.ieee.org/document/6600851)[7] which use reference directions and introduce additional computational complexity, we don't "pin" the solutions around reference vector, but still offer them certain opportunity to explore.

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
#### Optimized Perato Fronts

During the iteration, the hypervolume for perato front was used to track and visualize the performance of the three NSGA algorithms.

![[comp_hv.png]]

The graph clearly illustrates the superior performance of the fully optimized strategy, followed by the selection-optimized NSGA-II, while the standard NSGA-II trails behind.

From the visual representation:

1. **Initial Phase**: Both the standard NSGA-II and the selection-optimized variant start with nearly identical hypervolume values, suggesting a similar initial performance. In stark contrast, the fully optimized strategy — with its tailored initialization — begins its journey with a significantly enhanced hypervolume, indicating a promising outset.
    
2. **Evolutionary Progression**: The standard NSGA-II showcases two prominent increase in its hypervolume, signifying two key moments when its Pareto front witnessed substantial advancements. On the other hand, the two optimized versions depict more consistent and frequent upticks in their hypervolumes. This regularity in growth accentuates the effectiveness of the refined tournament selection strategy, affirming its role in directing the population's exploration of the solution space more proficiently.
    
3. **Convergence and Duration**: The algorithm equipped with dynamic mutation parameters exhibits a decelerated convergence rate, leading to an extended training duration. However, its hypervolume continues to expand, especially as it nears convergence. This persistence in growth underscores the advantage of dynamic mutation, allowing the algorithm to continually refine and potentially discover superior solutions even in the latter stages of the optimization process.

|2d plot|3d plot (with sharpe ratio)|
| ----------- | ----------- |
| ![[comp_solution.png]]| ![[comp_sharpe.png]]       |

This graph illustrates the final solution sets produced by all three algorithms, highlighting the distribution of the Pareto front determined by each. The diagram effectively showcases the advancements in selection optimization, dynamic mutation parameters, and initialization optimization.

When compared to the Pareto front of the standard NSGA-II, selection optimization offers a Pareto front with superior distribution and a broader coverage of all potential solutions. The standard version not only lacks solutions for portfolios with extreme returns but also offers limited optimization for solutions not situated at the center. However, in the standard NSGA-II's favor, a high population density in the middle indicates that this narrow region has been thoroughly explored, meaning its solutions within this range outperform the selection optimization algorithm.

The introduction of dynamic mutation parameters and initialization optimization nudges the Pareto front closer to the ideal direction. In the fully optimized version, the entire front outperforms the other two algorithms. This implies that for every solution found by the other algorithms, the fully optimized algorithm has identified solutions that are superior in both return and volatility, achieving complete dominance. Consequently, as shown in the plot with sharpe ratio, the optimized algorithm gives a higher value.
#### MCMC Projection

The Monte Carlo Markov Chain (MCMC) is a statistical method that uses the principles of Markov chains to sample from a probability distribution. This sampling technique is particularly useful for dimensions that are difficult to navigate, making it a robust tool in forecasting complex systems.

In portfolio optimization, a Pareto front that offers comprehensive coverage over the solution space, catering to both aggressive and conservative investors is not the only dimention we aiming at. Additionally, we seek the singular best portfolio considering both return and volatility. So, for each algorithm, we choose the solution with the highest Sharpe ratio as the optimal one and employ MCMC to forecast its performance over the next ten years for comparison.

![[comp_mcmc.png]]

The box plot displays the projected returns using MCMC for the next ten years. From the diagram, it's evident that the selection-optimized solution excels in terms of max return, min return, majority, and average. Furthermore, the fully optimized version surpasses even that.

##### NSGA-II
| Year |   < 0  |   > 0  |  >= 1  |  >= 2  |  >= 3  | >= 5  | >= 10 |
|------|--------|--------|--------|--------|--------|-------|-------|
|    1 |  30.86 |  69.14 |   1.87 |   0.03 |   0.00 |  0.00 |  0.00 |
|    2 |  23.34 |  76.66 |  13.09 |   1.25 |   0.13 |  0.00 |  0.00 |
|    3 |  18.54 |  81.46 |  26.84 |   6.34 |   1.71 |  0.11 |  0.00 |
|    4 |  15.80 |  84.20 |  39.07 |  14.99 |   5.87 |  0.87 |  0.03 |
|    5 |  12.77 |  87.23 |  49.45 |  23.97 |  12.20 |  3.05 |  0.12 |
|    6 |  10.52 |  89.48 |  57.75 |  33.56 |  19.33 |  6.53 |  0.65 |
|    7 |   8.93 |  91.07 |  64.07 |  42.44 |  27.35 | 12.02 |  1.83 |
|    8 |   7.48 |  92.52 |  70.00 |  49.84 |  35.57 | 18.06 |  4.21 |
|    9 |   6.46 |  93.54 |  74.24 |  56.57 |  42.31 | 24.63 |  7.20 |
|   10 |   5.31 |  94.69 |  78.09 |  62.29 |  49.82 | 31.05 | 11.29 |

##### Selection Optimized
| Year |   < 0  |   > 0  |  >= 1  |  >= 2  |  >= 3  |  >= 5  | >= 10 |
|------|--------|--------|--------|--------|--------|-------|-------|
|    1 |  30.46 |  69.54 |   2.73 |   0.03 |   0.00 |   0.00 |  0.00 |
|    2 |  23.65 |  76.35 |  15.67 |   2.11 |   0.22 |   0.01 |  0.00 |
|    3 |  18.70 |  81.30 |  29.55 |   8.39 |   2.39 |   0.22 |  0.00 |
|    4 |  15.61 |  84.39 |  41.40 |  17.75 |   7.70 |   1.49 |  0.05 |
|    5 |  12.47 |  87.53 |  51.46 |  26.53 |  14.57 |   4.50 |  0.24 |
|    6 |  10.62 |  89.38 |  59.24 |  36.40 |  22.46 |   8.96 |  1.27 |
|    7 |   8.56 |  91.44 |  65.98 |  44.54 |  30.12 |  14.45 |  3.11 |
|    8 |   7.29 |  92.71 |  71.42 |  51.98 |  37.70 |  20.44 |  5.74 |
|    9 |   5.94 |  94.06 |  75.89 |  58.53 |  44.94 |  27.43 |  9.54 |
|   10 |   5.19 |  94.81 |  79.84 |  64.42 |  51.68 |  34.09 | 14.19 |

##### Fully Optimized
| Year |   < 0  |   > 0  |  >= 1  |  >= 2  |  >= 3  |  >= 5  | >= 10 |
|------|--------|--------|--------|--------|--------|-------|-------|
|    1 |  25.27 |  74.73 |   3.17 |   0.05 |   0.00 |   0.00 |  0.00 |
|    2 |  17.39 |  82.61 |  19.73 |   3.16 |   0.47 |   0.01 |  0.00 |
|    3 |  13.25 |  86.75 |  38.20 |  12.05 |   3.97 |   0.43 |  0.00 |
|    4 |   9.97 |  90.03 |  52.33 |  25.09 |  11.76 |   2.54 |  0.16 |
|    5 |   7.30 |  92.70 |  63.00 |  37.92 |  21.98 |   7.37 |  0.70 |
|    6 |   5.74 |  94.26 |  72.39 |  49.13 |  32.73 |  14.83 |  2.67 |
|    7 |   4.61 |  95.39 |  78.30 |  59.14 |  43.23 |  23.66 |  6.34 |
|    8 |   3.38 |  96.62 |  82.90 |  67.02 |  52.30 |  32.17 | 11.24 |
|    9 |   2.68 |  97.32 |  86.85 |  73.78 |  61.00 |  41.47 | 17.65 |
|   10 |   2.09 |  97.91 |  89.20 |  78.77 |  68.14 |  50.53 | 24.46 |

The three tables generated by MCMC further underscore the complete dominance of the optimized algorithm's returns over the standard NSGA-II algorithm. For each year in the upcoming decade, the optimized solution presents a higher likelihood of yielding superior returns.

## Conclusion

Over the course of this study, we first revisited the foundational principles of portfolio optimization and delved into the nuances of Multi-objective Evolutionary Algorithms, specifically the NSGA-II. Our primary goal was to optimize performance of the original NSGA-II algorithm, by applying a set of strategic modifications. Our proposed enhancements aimed to catalyze individuals' exploration of the solution space more intelligently and effectively. Through rigorous testing and comparison, particularly using the MCMC for forecasting over a decade, we observed marked improvements in the modified NSGA-II's efficiency.

Additionally, some interesting topics remain unexplored regarding our enhanced NSGA-II. Comparative performance analyses with NSGA-III or even DNN-based methods in portfolio optimization would be insightful. Moreover, assessing its efficacy across a variety of multi-objective optimization tasks also warrants attention.

In summary, while many advancements like NSGA-III have exhibited superior capabilities, the potential of the original NSGA-II remains vast. By tweaking its rules and functions, we've illustrated the algorithm's potential in navigating the intricacies of modern portfolio optimization. 
## References

Bibliography

[1]    K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist multiobjective genetic algorithm: NSGA-II,” _IEEE Transactions on Evolutionary Computation_, vol. 6, no. 2, pp. 182–197, Apr. 2002, doi: https://doi.org/10.1109/4235.996017.

[2]    G.-Y. Ban, N. El Karoui, and A. E. B. Lim, “Machine Learning and Portfolio Optimization,” _Management Science_, vol. 64, no. 3, pp. 1136–1154, Mar. 2018, doi: https://doi.org/10.1287/mnsc.2016.2644.

[3]    Z. Zhang, S. Zohren, and S. Roberts, “Deep Learning for Portfolio Optimization,” _The Journal of Financial Data Science_, vol. 2, no. 4, pp. 8–20, Aug. 2020, doi: https://doi.org/10.3905/jfds.2020.1.042.

[4]    R. Mansini, W. Ogryczak, and M. G. Speranza, “Twenty years of linear programming based portfolio optimization,” _European Journal of Operational Research_, vol. 234, no. 2, pp. 518–535, Apr. 2014, doi: https://doi.org/10.1016/j.ejor.2013.08.035.

[5]    S. Dhanalakshmi, S. Kannan, K. Mahadevan, and S. Baskar, “Application of modified NSGA-II algorithm to Combined Economic and Emission Dispatch problem,” _International Journal of Electrical Power & Energy Systems_, vol. 33, no. 4, pp. 992–1002, May 2011, doi: https://doi.org/10.1016/j.ijepes.2011.01.014.

[6]    Rio D’Souza, K. Chandra Sekaran, and A. Kandasamy, “Improved NSGA-II Based on a Novel Ranking Scheme,” Feb. 2010, doi: https://doi.org/10.48550/arxiv.1002.4005.

[7]    K. Deb and H. Jain, “An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints,” _IEEE Transactions on Evolutionary Computation_, vol. 18, no. 4, pp. 577–601, Aug. 2014, doi: https://doi.org/10.1109/tevc.2013.2281535.

[8]    Haitham Seada and K. Deb, “U-NSGA-III: A Unified Evolutionary Optimization Procedure for Single, Multiple, and Many Objectives: Proof-of-Principle Results,” pp. 34–49, Jan. 2015, doi: https://doi.org/10.1007/978-3-319-15892-1_3.

[9]    Q. Liu, X. Liu, J. Wu, and Y. Li, “An Improved NSGA-Ⅲ Algorithm Using Genetic K-means Clustering Algorithm,” _IEEE Access_, pp. 1–1, 2019, doi: https://doi.org/10.1109/access.2019.2960531.

[10]    Z.-M. Gu and G.-G. Wang, “Improving NSGA-III algorithms with information feedback models for large-scale many-objective optimization,” _Future Generation Computer Systems_, vol. 107, pp. 49–69, Jun. 2020, doi: https://doi.org/10.1016/j.future.2020.01.048.

[11]    B. Doerr and Z. Qu, “Runtime Analysis for the NSGA-II: Provable Speed-Ups from Crossover,” vol. 37, no. 10, pp. 12399–12407, Jun. 2023, doi: https://doi.org/10.1609/aaai.v37i10.26461.

[12]    B. Doerr and Z. Qu, “From Understanding the Population Dynamics of the NSGA-II to the First Proven Lower Bounds,” vol. 37, no. 10, pp. 12408–12416, Jun. 2023, doi: https://doi.org/10.1609/aaai.v37i10.26462.

[13]    X. Chu and X. Yu, “Improved Crowding Distance for NSGA-II,” Nov. 2018, doi: https://doi.org/10.48550/arxiv.1811.12667.