---
layout: post
title:  "CVaR-optimal allocation for liquidity providers with a chance constraint"
date:   2024-03-16
categories: defi
---

In this article, we explore an algorithm designed for liquidity providers (LPs) problem looking for an optimal allocation of the funds between the liquidity pools with varying trading characteristics. The algorithm uses a simulated DeFi trading environment to generate returns distributions, and a gradient-descent iteration to find the CVaR-optimal (conditional value-at-risk) allocation of the funds subject to certain profitability of the strategy, expressed as a probabilistic constraint. In order to to account for the market impact of an individual LP, we alternate between the simulation and gradient descent steps.

The algorithm was part of the $2^{\text{nd}}$ place submission in the [SIAG/FME Code Quest 2023](https://sites.google.com/view/siagfme-codequest2023/challenge) data challenge we took part together with colleagues from HU Berlin and University of Oxford, organized by [SIAM Activity Group on Mathematical Finance and Engineering (SIAG/FME)](https://www.siam.org/membership/activity-groups/detail/financial-mathematics-and-engineering) and sponsored by [World Business Chicago](https://worldbusinesschicago.com/).

> The code for this article is available on [github](https://github.com/starovoitovs/defi). The code provides separate modules for the simulated trading environment and the allocation algorithm.

## Introduction

In conventional finance, markets are characterized by trading in order book and OTC trades. Order books facilitate the submission of diverse orders by traders, specifying price, volume, and trade intentions, managed by financial institutions and trusted intermediaries. OTC markets, conversely, operate through a network of financial entities and market makers to enable direct securities transactions.

In contrast, DeFi represents a decentralized ecosystem of blockchain-based financial services, bypassing intermediaries such as brokers and banks. Utilizing blockchain as a peer-to-peer network, users can deploy and share immutable smart contracts, autonomously governing interactions.

Within the DeFi universe, automated market makers (AMM) emerge as protocols facilitating trades between distinct coins, akin to Bitcoin and Ethereum. AMM enable traders to execute trades, provide liquidity to a shared asset 'pool', or withdraw assets as needed. Predominantly, AMM operate as constant product market makers (CPMM), wherein liquidity providers receive LP tokens representing their share of the pool upon deposit and can withdraw their assets by burning these LP tokens.

## Constant product market making

CPMMs operate according to specific rules dictating the interactions between liquidity takers (LT) and liquidity providers (LP) within the pool, known as the **LT trading condition** and the **LP trading condition**. In the case of LTs, these rules, alongside the pool's available reserves $(R^X, R^Y)$, govern the mechanics of coin swapping. Conversely, for LP, the LP trading condition governs the amount of coins X and Y they deposit when adding liquidity or withdraw when removing it from the pool.

### LT trading condition

LT transactions involve swapping a quantity $y$ of $\mathcal Y$ for a quantity $x$ of $\mathcal X$, and vice-versa. The LT trading condition links the state of the pool before and after a swap is executed. The condition requires that the product of the reserves in both coins remains constant before and after the transaction.

In practice, CPMMs charge a proportional fee $\phi$ to LTs. More precisely, when the swap of an LT is for a quantity $x$ in the pool, then the swap is executed for a quantity $x(1-\phi)$. Similarly, when the swap of an LT is for a quantity $y$ in the pool, the swap is executed for a quantity $y(1-\phi)$.

Below, we describe this more precisely.

#### Swap $\mathcal X$ to $\mathcal Y$

When a trader deposits $x$ of $\mathcal X$ they receive $y$ of $\mathcal Y$. The amount $y$ of $\mathcal Y$ that they receive is determined s.t.

$$
    R^X R^Y=\left(R^X+(1-\phi) x\right)\left(R^Y-y\right) \quad \Longrightarrow \quad y=x \frac{(1-\phi) R^Y}{R^X+(1-\phi) x}.
$$

Even though the trade is executed for a proportion $(1-\phi) x$ of the quantity sent to the pool, the all $x$ of $\operatorname{coin}-X$ are still put in the pool. Therefore, after the swap event the reserves in the pool are updated according to

$$
    R^X \longrightarrow R^X+x \quad \text{ and } \quad R^Y \longrightarrow R^Y-y.
$$

#### Swap $\mathcal Y$ to $\mathcal X$

When a trader deposits $y$ of $\mathcal Y$ they receive $x$ of $\mathcal X$. The amount $x$ of $\mathcal X$ that they receive is determined s.t.

$$
    R^X R^Y=\left(R^Y+(1-\phi) y\right)\left(R^X-x\right) \quad \Longrightarrow \quad x=y \frac{(1-\phi) R^X}{R^Y+(1-\phi) y}.
$$

Similar to the previous case, the reserves of the pool are updated according to

$$
    R^X \longrightarrow R^X-x \quad \text{ and } \quad R^Y \longrightarrow R^Y+y.
$$

#### Marginal price and execution cost

It is worthwhile investigating the above relationships when the amount being swapped is very small. To this end, note that for an infinitesimal quantity $y \ll 1$ of $\mathcal Y$ swapped, the amount $x$ of $\mathcal X$ the trader receives is

$$
    x=y \frac{R^X}{R^Y+y}=y \frac{R^X}{R^Y}\left(1+\frac{y}{R^Y}\right)^{-1}=y \frac{R^X}{R^Y}\left(1-\frac{y}{R^Y}+o\left(\frac{y}{R^Y}\right)\right)
$$

where the last equality follows from the Taylor expansion $(1+a)^{-1}=1-a+o(a)$. Therefore, the ratio of $x$ to $y$ has limit

$$
    \lim _{y \rightarrow 0} \frac{x}{y}=\frac{R^X}{R^Y} .
$$

The ratio $\frac{R^X}{R^Y}$ is referred to as marginal price of the pool, which is akin to the midprice in traditional markets. The marginal price is a reference price - the difference between its value and the actual ratio $\frac{x}{y}$ is called the execution cost.

### LP trading condition

LPs interact with the pool by submitting both coins into the pool (called $\texttt{minting}$ liquidity) and receiving liquidity providing coins (LP tokens). The amount of these LP tokens they receive reflects the percentage of liquidity they have provided to the pool – the precise definition is below. LPs may subsequently return LP tokens and receive a certain amount of $\mathcal X$ and $\mathcal Y$. This act is called $\texttt{burning}$ liquidity, and the precise amount they receive is also described below.

In either LP event mint or burn, the LP trading condition requires that LP operations do not impact the marginal price $\frac{R^X}{R^Y}$.

#### Minting

More precisely, when the LP wishes to mint LP tokens, the quantities of x and y they submit must respect the equality of marginal price before and after the mint event, so that x and y must satisfy the relationship:

$$
    Z=\frac{R^X}{R^Y}=\frac{R^X+x}{R^Y+y} \quad \Rightarrow \quad \frac{x}{y}=\frac{R^X}{R^Y}
$$

Upon submitting the correct amount of coins, the LP trader will then receive LP tokens in the amount equal to:

$$
    \ell=\frac{x}{R^X} L=\frac{y}{R^Y} L
$$

where $L$ is the outstanding amount of LP tokens issued by the pool prior to trader's LP mint event. Further, after the LP mint event the reserves and outstanding LP tokens are updated as follows

$$
    R^X \rightarrow R^X+x, \quad R^Y \longrightarrow R^Y+y, \quad \text { and } \quad L \longrightarrow L+\ell
$$

#### Burning

Similarly, when the trader wishes to burn LP coins, the quantities of x of $\mathcal X$ and y of $\mathcal Y$ they receive respects the equality of marginal price before and after the burn event. As well, they receive an amount equal to proportion of LP coins they hold relative to the total outstanding coins. In particular, they receive the following amount of $\mathcal X$ and $\mathcal Y$ :

$$
    x=\frac{\ell}{L} R^X \quad \text { and } \quad y=\frac{\ell}{L} R^Y,
$$

where all quantities on the right hand side of the equalities are the quantities just prior to the burn event. Furthermore, after the textttLP burn event, the the reserves and outstanding LP coins are updated as follows

$$
    R^X \rightarrow R^X-x, \quad R^Y \longrightarrow R^Y-y, \quad \text { and } \quad L \rightarrow L-\ell.
$$

## Optimization problem for LP

After the brief introduction into the mechanics of DeFi, we will elaborate on the optimization problem for the LP.

Participants are provided with a fixed initial state of $N_{pools}$ pools characterized by $(R^X_0, R^Y_0, \phi)$, and start with the initial holdings of $x_0$ of $\mathcal X$ coin. The objective is to compute the amounts (in percentage of $x_0$) to put in each of the $N_{pools}$ pools.

### Trading environment

We are given a fixed trading environment defined by the tuple $(\kappa, p, \sigma, T)$. The dynamics of the swaps in the trading enviroment follows a jump process, where the trades can take place either in each pool individually, or in all pools simultaneously -- leading to correlations between the pools. 

The parameters are interpreted as follows:

* $\kappa \in \mathbb{R}^{N_{pools} + 1}$ corresponds to the jump intensity of the underlying Poisson process. The extra coordinate corresponds to the parameter value in the event of simultaneous jumps.
* $p\in \mathbb{R}^{N_{pools} + 1}$ corresponds to the directionality of the trade: with probability $p$ one swaps on average 1 unit of $\mathcal X$ to $\mathcal Y$, and with probability $1-p$ one swaps $R^Y/R^X$ units of $\mathcal Y$ to $\mathcal X$, so that in each trade on average 1 unit of $\mathcal X$ is traded. The extra coordinate corresponds to the parameter value in the event of simultaneous jumps.
* $\sigma\in \mathbb{R}^{N_{pools} + 1}$ corresponds to the volatility of the trade. The extra coordinate corresponds to the parameter value in the event of simultaneous jumps.
* $T\in\mathbb R$ corresponds to the timeframe.

### Problem setup

The amounts should optimize the following performance criterion:

$$
    \begin{align}
        \label{eq: cvar objective}
        \min_w &\quad \text{CVaR}_\alpha (r_T) \\
        \label{eq: chance constraint}
        &\mathbb{P}(r_T > \zeta) > q
    \end{align}
$$

where $r_T=\log \left(x_T\right)-\log \left(x_0\right)$ is the final performance of the LP, $\theta$ is the distribution of the initial wealth across the $n$ pools and $\alpha, \zeta$, and $q$ are in the file params.py. The conditional Value-at-Risk (also called expected shortfall) at level $\alpha \in(0,1)$ for a variable $r$ (regarded as the performance of a portfolio) is

$$
    \operatorname{CVaR}(r)=\frac{1}{1-\alpha} \int_\alpha^1 \operatorname{VaR}_s(r) d s \approx \mathrm{E}\left[-r \mid-r \geq \operatorname{VaR}_\alpha\right],
$$

where $\operatorname{VaR}_s(r)$ is the Value-at-Risk at level $s \in(0,1)$ of a real-valued random variable $r$ (regarded as the performance of a portfolio) is

$$
    \operatorname{VaR}_\alpha(r)=-\inf \{z \in \mathbb{R} \mid \mathbb{P}(r \leq z) \leq 1-\alpha\},
$$

i.e., $\operatorname{VaR}_\alpha(r)$ is the $\alpha$-quantile of $-r$. Note that $\mathrm{VaR}$ can be computed using the standard numpy/Python quantile function numpy .quantile function. Respectively, CVaR can be approximated using the formula above as sample conditional mean.

## Methodology and algorithm

Essentially, our algorithm uses gradient descent on the paths generated by the simulated environment to find the optimal weights, and in order to account for the market impact, we alternate between market generation and gradient descent.

Main loop consists of $N_{iter}$ iterations, where each of these iterations includes a market generation step and a gradient descent loop. At the beginning of each iteration $i\in\\{1, \dots, N_{iter}\\}$ we use the weights $w_i$ obtained from the last iteration to generate the market paths, and within this iteration we use these paths in the gradient descent iteration of $N_{GD}$ steps to obtain new weights $w_{i+1}$. The idea behind alternation between market generation and gradient descent is that once the weights have changed, the market impact of the pool allocations has changed as well, which affects profitability of the pools.

In order to account for this, at each step $j\in\\{1,\dots,N_{GD}\\}$ of the gradient descent iteration we reweigh the holdings $\overline x$, $\overline y$ using the weight iterates $\widetilde w_j$, and use the characteristics of the best pool $(\overline R^X, \overline R^Y, \overline \phi) \in (\mathbb{R}^{N_{batch}})^3$ -- which will be different for each $k\in\\{1,\dots,N_{batch}\\}$ -- to compute the ultimate holdings denominated in $\mathcal X$ after the exchange of $\mathcal Y$ into $\mathcal X$ in the best pool. This allows us to compute returns for each path $k\in\{1,\dots,N_{batch}\}$, which we will use to compute objectives in the optimization routine.

### Algorithm description

The algorithm proceeds as follows. At the beginning of each step $i\in\{1, \dots, N_{iter}\}$, we start with some weights $w_i\in\mathbb{R}^{N_{pools}}$. We then invoke the function

$$
    \texttt{GenerateMarket}(w_i) = \left(\overline{x}, \overline{y}, \overline R^X, \overline R^Y, \overline{\phi}\right),
$$

which generates $N_{batch}$ paths of the market evolution starting from initial pool allocation $x_0 \cdot w_i$, such that $\overline{x}, \overline{y} \in \mathbb{R}^{N_{batch} \times N_{pools}}$ correspond to the amounts of $\mathcal X$, $\mathcal Y$ in each pool obtained after burning the liquidity tokens at the end of the period; $\overline R^X, \overline R^Y\in\mathbb{R}^{N_{batch}}$ correspond to the stock of the $\mathcal X$, $\mathcal Y$ in the pool offering the best price; and $\overline{\phi}\in\mathbb{R}^{N_{batch}}$ corresponds to the fee parameter of the pool offering the best price.

Thereafter, in the $i$-th iteration of the main loop we enter a gradient descent loop of $N_{GD}$ steps, which yields the next weight iterate $w_{i+1}$ based on the generated paths. Specifically, for $j\in\{1,\dots,N_{GD}\}$, in the $j$-th step of the gradient descent loop, we consider the weight iterate $\widetilde w_j \in \mathbb{R}^{N_{pools}}$ and approximate the returns of the reweighed portfolios by

$$
    \begin{equation}
    \label{eq:returns}
    \begin{aligned}
        \overline x_{burn} &:= \overline{x} \left(\widetilde w_j \odot w_i^{-1}\right) \in\mathbb{R}^{N_{batch}}, \\
        \overline y_{burn} &:= \overline{y} \left(\widetilde w_j \odot w_i^{-1}\right) \in\mathbb{R}^{N_{batch}}, \\
        \overline x_{swap} &:= \frac{\overline y_{burn} \odot (1-\overline{\phi}) \odot \overline R^X}{\overline R^Y + (1-\overline{\phi}) \odot \overline y_{burn}} \in\mathbb{R}^{N_{batch}}, \\
        \overline r &:= \log(\overline x_{burn} + \overline x_{swap}) - \log(x_0) \in\mathbb{R}^{N_{batch}},
    \end{aligned}
    \end{equation}
$$

where the inverse $w_i^{-1}$ and division are considered elementwise, and we denote elementwise multiplication by $\odot$. For $w\in\mathbb{R}^{N_{pools}}$ and $\overline r\in\mathbb{R}^{N_{batch}}$ we introduce the following loss functions

$$
    \begin{aligned}
    \ell_1(w, \overline r) &= \left(q - \frac{1}{N_{batch}} \sum_{i=1}^{N_{batch}} G(1000 \cdot (r_i - \zeta))\right)_+^2,\\
    \ell_2(w, \overline r) &= \frac{1}{N_{pools}} \sum_{i=1}^{N_{pools}} (-w_i)_+,\\
    \ell_3(w, \overline r) &= \left(\sum_{i=1}^{N_{pools}}w_i - 1\right)^2, \\
    \ell_4(w, \overline r) &= \operatorname{CVaR}_\alpha = \frac{\sum_{i=1}^{N_{batch}} (-r_i) 1_{\{-r_i > q_\alpha(-\overline r)\}}}{|\{i\in \{1,\dots,N_{batch}\}: -r_i > q_\alpha(-\overline r)\}|},
    \end{aligned}
$$

where $G$ is the sigmoid function $G(z) := \frac{1}{1 + e^{-z}}$, $(z)\_+ = z 1\_{\{z>0\}}$ is the ReLU function, and $q_\alpha: \mathbb{R}^{N_{batch}} \rightarrow \mathbb{R}$ denotes the empirical quantile function. We define the total loss function as

$$
    \ell(w,\overline r) := \theta_1 \ell_1(w,\overline r) + \theta_2 \ell_2(w,\overline r) + \theta_3 \ell_3(w,\overline r) + \theta_4 \ell_4(w,\overline r).
$$

where we used the loss weights $\theta$ which are used to ensure that the loss components are of the same order. The interpretation is straightforward: $\ell_1$ corresponds to the chance constraint \eqref{eq: chance constraint}, where we approximate $\mathbb{E}[1_{r>\zeta}]$ with sigmoid function $G$ to maintain differentiability of the loss; $\ell_2$ ensures that the weights are not negative; $\ell_3$ ensures that the weights sum to one; $\ell_4$ is the CVaR objective \eqref{eq: cvar objective} to be minimized. The gradient descent is implemented in \texttt{pytorch} with gradients computed by backpropagation. Note that our algorithm crucially uses the fact that \texttt{torch.quantile} function, which is used for the CVaR implementation, is differentiable.

We present in pseudocode the algorithm below. Note that $\log$, division and inverse are taken elementwise, whenever applicable. Also note that in the algorithm we make small adjustments which we don't include in the pseudocode, such as flooring weights at $0$ to ensure that they are non-negative, adding a small number $\varepsilon = 0.0001$ to each weight to ensure that they are not zero, and normalize them so that they sum up to $1$. This is necessary, since these constraints may be violated. However, the violation is being minimized due to being encoded in the loss.

![Algorithm](/assets/posts/defi-cvar/algorithm.png)
*Algorithm in pseudocode.*

## Choice of the initial value

In order to obtain a good initial value for the weights, we consider profitability of the respective pools in isolation, where we invest all initial stock in a single pool. That is, for each $i\in\{1,\dots,N_{pools}\}$ we generate returns according to the formula \eqref{eq:returns} with initial weights $w^i\in\mathbb{R}^{N_{pools}}$ defined as

$$
    w^i_j = \begin{cases}
        1,& i = j \\
        0,& i\neq j.
    \end{cases}
$$

The profitability metrics of the pools, i.e. mean return, $\text{CVaR}_\alpha$ and $\mathbb{P}(r_T \geq \zeta)$ are displayed below.

![Pool metrics](/assets/posts/defi-cvar/pool_metrics.png)
*Mean return, $\text{CVaR}_\alpha$ and $\mathbb{P}(r_T \geq \zeta)$ whenever allocating all initial stock in a single pool $i$, for $i\in\{1,\dots,N_{pools}\}$.*

In the table below we summarize ranking of the pools across different metrics. It suggests that Pool 6 is the worst performing pool, Pool 1 and Pool 5 are slightly better, followed by Pool 4, while Pool 2 and Pool 3 are among the best performers.

![Pool ranking](/assets/posts/defi-cvar/pool_ranking.png)
*Ranking of the pools for individual performance metrics.*

This leads us to the choice of the initial allocation corresponding to the equal distribution between the best pools, i.e. initial values coming into consideration are:

$$
    w_{011000} := \left(0, \frac12, \frac12, 0, 0, 0\right),\qquad w_{011100} := \left(0, \frac13, \frac13, \frac13, 0, 0\right).
$$

The figure below indicates that initial weights $w_{011000}$ yield a better result for the seed \texttt{4294967143} in terms of the optimal CVaR. Meanwhile, initial weights $w_{011100}$ yield slightly better performance on average among other values of seeds, with difference in terms of optimal CVaR between $w_{011000}$ and $w_{011100}$ being of order $10^{-4}$.

![CVaR difference](/assets/posts/defi-cvar/cvar_difference.png)
*Difference $\text{CVaR}_\alpha^{011000} - \text{CVaR}_\alpha^{011100}$ for different seeds, obtained with different initial values $w_{011000}$ and $w_{011100}$.*

## Results

We consider the following values of the model parameters:

| Parameter     | Value                                               | Interpretation                              |
|---------------|-----------------------------------------------------|---------------------------------------------|
| $N_{pools}$   | $\texttt{6}$                                        | Number of pools                             |
| $N_{batch}$   | $\texttt{1000}$                                     | Number of paths                             |
| $R^X_0$       | $\texttt{[100, 100, 100, 100, 100, 100]}$           | Initial stock of $\mathcal X$ in each pool  |
| $R^Y_0$       | $\texttt{[1000, 1000, 1000, 1000, 1000, 1000]}$     | Initial stock of $\mathcal Y$ in each pool  |
| $\phi$        | $\texttt{[0.03, 0.03, 0.03, 0.03, 0.03, 0.03]}$     | Proportional fee in each pool               |
| $x_0$         | $\texttt{10}$                                       | Initial holdings of the LP                  |
| $\alpha$      | $\texttt{0.9}$                                      | CVaR quantile                               |
| $q$           | $\texttt{0.8}$                                      | Profitability threshold                     |
| $\zeta$       | $\texttt{0.05}$                                     | Profitability quantile                      |
| $\kappa$      | $\texttt{[0.25, 0.5, 0.5, 0.45, 0.45, 0.4, 0.3]}$   | Jump intensity                              |
| $\sigma$      | $\texttt{[1., 0.3, 0.5, 1., 1.25, 2, 4]}$           | Jump volatility                             |
| $p$           | $\texttt{[0.45, 0.45, 0.4, 0.38, 0.36, 0.34, 0.3]}$ | Directionality of swaps                     |
| $T$           | $\texttt{60}$                                       | Timeframe                                   |

Furthermore, we consider the following numerical parameters

| Parameter                                      | Value                                                 | Interpretation                        |
|------------------------------------------------|-------------------------------------------------------|---------------------------------------|
| $\beta$                                        | $\texttt{0.01}$                                       | Learning rate of the GD               |
| $(\theta_1, \theta_2, \theta_3, \theta_4)$     | $\texttt{[10, 1, 1, 1]}$                              | Loss weights                          |
| $w$                                            | $\texttt{[0.001, 0.332, 0.332, 0.332, 0.002, 0.001]}$ | Initial weights                       |
| $N_{iter}$                                     | $\texttt{20}$                                         | Number of iterations in the main loop |
| $N_{GD}$                                       | $\texttt{2000}$                                       | Number of steps in the GD iteration   |

### Results

With these parameters, we attain a CVaR of $\texttt{0.00376146}$, which is very close to the optimal one obtained by the grid search. The optimal allocation between the pools is given by

$$
    \texttt{[0.13075465, 0.24796906, 0.22088264, 0.14375485, 0.23937343, 0.0172653]}
$$

We observe that the optimal allocation confirms our heuristic for the choice of the initial value in the previous section.

### Initial value

We executed the algorithm for 3 different starting values $w$, for 30 different values of seed ranging from $4294967143$ to $4294967162$, with $N_{iter}=20$ iterations of the main loop, each encompassing $N_{GD}=2000$ of the gradient descent steps with learning rate $\beta=0.01$. For the initial values we considered

$$
    w_{011100} := \left(0, \frac13, \frac13, \frac13, 0, 0 \right),\qquad w_{000001} := \left(0, 0, 0, 0, 0, 1 \right),\qquad w_{111111} := \left(\frac16, \frac16, \frac16, \frac16, \frac16, \frac16\right).
$$

The value $w_{011100}$ corresponds to the heuristic described in the previous section. The value $w_{111111}$ corresponds to the uniformly distributed portfolio, which one would start from without any additional information about the dynamics of the system, and we demonstrate that our algorithm delivers good results for this starting point as well (but not as good as our educated guess). The value $w_{000001}$ was chosen as a bad initial guess, which violates the chance constraint and attains rather large CVaR value, and we demonstrate that our algorithm yields roughly the same results as for the starting point $w_{111111}$, despite a rather bad initial guess.

### Convergence behavior

Convergence of the CVaR objective as a function of iteration is displayed in the figure below. We observe that the algorithm already provides a very good candidate after 1 iteration (even for the bad initial guess $w_{000001}$), and converges after roughly 5 iterations, while attaining the minimum after the first iteration for roughly half of the paths.

![Convergence](/assets/posts/defi-cvar/convergence.png)
*Convergence of the CVaR for different seeds as function of iteration for $w_{011100}$ (left), $w_{000001}$ (right) and $w_{111111}$ (bottom).*

### Chance constraint

Behavior of the chance constraint as a function of iteration is displayed in the figure below. We observe that for those paths that satisfied the constraint from the beginning, the algorithm never violates it again, and for those which violated the constraint in the beginning, the algorithm moves swiftly towards the barrier $q = 0.8$, which for the bad initial value $w_{000001}$ happens right away after 1 iteration. Concerning the paths which never cross the barrier $q=0.8$ and always stay below it, in all likelihood samples corresponding to those seeds do not admit enough profitability in the first place.

![Chance constraint](/assets/posts/defi-cvar/constraint.png)
*Convergence of chance constraint $\mathbb{P}(r\geq \zeta)$ for different seeds as function of iteration for $w_{011100}$ (left), $w_{000001}$ (right) and $w_{111111}$ (bottom), with the threshold value $q=0.8$.*

### Pool allocation

Pool allocation among different values of the seeds is displayed in the figure below. We note that the allocation is relatively stable between the seeds, and would be more homogeneous if one would consider larger values of $N_{batch}$.

![Allocation](/assets/posts/defi-cvar/allocation.png)
*Optimal allocation among the pools for different seeds as function of iteration for $w_{011100}$ (left), $w_{000001}$ (right) and $w_{111111}$ (bottom).*

### Performance

Comparison of CVaR values between different initial values $w_{011100}$, $w_{000001}$ and $w_{111111}$ is displayed in the figure below, i.e. for each seed we display

$$
    \operatorname{CVaR}_{000001} - \operatorname{CVaR}_{011100},\qquad     \operatorname{CVaR}_{111111} - \operatorname{CVaR}_{011100},
$$

so that positive values indicate that the initial guess $w_{011100}$ is better than the other. We observe that our chosen heuristic for the initial value beats other initial values across most seeds.

![CVaR Performance](/assets/posts/defi-cvar/cvar_performance.png)
*Comparison of CVaR values attained for different seeds with different initial weights.*

## Running time

One iteration of the algorithm takes roughly 2--3 seconds on the M1 processor. Majority of this time (more than 2 seconds) is used to generate paths of the market. One step of the gradient descent takes roughly 50$\mu$s, and we are making $N_{GD}=2000$ gradient descent steps for each iteration of the main loop. Therefore, each iteration of the algorithm (market generation + gradient descent) is taking roughly 2--3 seconds.

The plots above suggest that after 1 iteration one already gets a very good candidate for the weights. At this point, if the speed is highest priority, one can just use $N_{iter}=1$ and obtain a reliable result after 2--3 seconds, otherwise one can obtain improvement of the CVaR of order $10^{-3}$ at the expense of some other 10 seconds.

# References

* SIAG/FME Code Quest 2023 [Manual](https://sites.google.com/view/siagfme-codequest2023/challenge).