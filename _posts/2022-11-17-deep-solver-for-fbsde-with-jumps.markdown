---
layout: post
title:  "Deep solver for FBSDE with jumps"
date:   2022-11-17
categories: machine-learning
comments: true
---

In this article we elaborate on the deep solver we came up with the colleagues from HU Berlin for the solution of the stochastic control problem including jumps, as part of the [Helmholtz GPU Hackathon 2022](https://www.aicampus.berlin/event/helmholtz-gpu-hackathon-2022). Generally, any stochastic control problem can be rewritten as a forward-backward SDE (FBSDE), presenting an alternative to the dynamic programming approach with Hamilton-Jacobi-Bellman equations. Such systems are widespread in mathematical finance, arising in pricing of contingent claims, risk management problems and calculations of value adjustments (xVA) to account for the counterparty risk. In this article, we extend the [deep FBSDE solver](https://link.springer.com/article/10.1007/s40304-017-0117-6) by E, Han and Jentzen, by making a deep ansatz for the control process $R$ corresponding to jumps, by analogy with the deep ansatz by E et al. for the control process $Z$ corresponding to the diffusive part.

## Stochastic control problem

We look at the stochastic control problem, where we denote forward dynamics by $X_t$ and control process by $u_t$. In our setup, the forward dynamics involves jumps and is written as follows: 

$$
d X_t=b\left(t, X_t, u_t\right) d t+\sigma\left(t, X_t, u_t\right) d B_t+\int_{\mathbb{R}^n} \gamma\left(t, X_{t-}, u_{t-}, w\right) \tilde{N}(d t, d w)
$$

The objective to be maximized is given by

$$
J(u)=\mathbb{E}\left[\int_0^T F\left(t, X_t, u_t\right) d t+G\left(X_T\right)\right]
$$

We want to employ probabilistic approach to the solution of this control problem. The Hamiltonian of the system is given by

$$
H(t, x, y, z, u, r)=b(x, u)^T y+\operatorname{tr}\left(\sigma \sigma^T z\right)+F(t, x, u)+\sum_{j=1}^{\ell} \sum_{i=1}^d \int_{\mathbb{R}} \gamma_{i j}\left(t, x, u, w_j\right) r_{i j}(t, w) \nu_j\left(d w_j\right)
$$

We introduce the adjoint process $Y_t$:

$$
\left\{\begin{array}{l}
d Y_t=-\nabla_x H\left(t, X_t, u_t, Y_t, Z_t, R_t\right) d t+Z_t d B_t+\int_{\mathbb{R}^n} R\left(t^{-}, w\right) \tilde{N}(d t, d w) \\
Y_T=\nabla_x G\left(X_T\right)
\end{array}\right.
$$

Intuitively, the components $Z$ and $R$ are there to "subtract the right amount of randomness", so that the terminal condition can be satisfied. According to the *classical Pontryagin's maximum principle*, the optimal control then extremizes the Hamiltonian, i.e. admissible control $\hat u$ is *optimal* if

$$
H\left(t, \hat{X}_t, \hat{u}_t, \hat{Y}_t, \hat{Z}_t, \hat{R}_t\right)=\sup _{u \in U} H\left(t, \hat{X}_t, u, \hat{Y}_t, \hat{Z}_t, \hat{R}_t\right).
$$

Importantly, we assume that the optimal control $\hat u_t$ can be written as $\operatorname{argmax}$ of the Hamiltonian above, which allows us to abstract out the control process $u_t$ completely, and we are left with the forward-backward system in $X, Y$ with control processes $Z, R$.

## FBSDE formulation

Rewritten in abstract terms, to the above control problem is given by the solution of the following forward-backward system:

$$
\begin{align}
\label{eq: fbsde}
\mathrm{d} X_t^{Y_0, Z, R}&=b\left(t, X_t^{Y_0, Z, R}, Y_t^{Y_0, Z, R}, Z_t\right) \mathrm{d} t+\sigma\left(t, X_t^{Y_0, Z, R}, Y_t^{Y_0, Z, R}, Z_t\right) \mathrm{d} W_t  \\
\nonumber &\qquad +\int_{\mathbb{R}^{\ell}} \gamma\left(t, X_t^{Y_0, Z, R}, Y_t^{Y_0, Z, R}, Z_t, w\right) N(\mathrm{d} t, \mathrm{d} w) \\[10pt]
\nonumber \mathrm{d} Y_t^{Y_0, Z, R}&=-f\left(t, X_t^{Y_0, Z, R}, Y_t^{Y_0, Z, R}, Z_t\right) \mathrm{d} t+Z_t \mathrm{d} W_t+\int_{\mathbb{R}^{\ell}} R_t N(\mathrm{d} t, \mathrm{d} w) \\[10pt]
\nonumber X_0&=x_0, \qquad Y_T = g(X_T) 
\end{align}
$$

We use the superscript $Y_0, Z, R$ to emphasize that we interpret the scalar $Y_0$ and processes $Z$ and $R$ as controls to be learned by the neural network. In this regard, the setup is similar to the solver of E et al. we referenced in the beginning, with the difference that now we also have the control process $R$ accounting for the jumps.

Note that our system is a *forward-backward* system, which means that we know the initial value of the forward process $X_t$ and terminal value of the backward process $Y_t$. Thus, following the idea from E et al., we set some random initial value for the initial value $Y_0$ of the backward process, and treat it as a trainable parameter, which will be learned by backpropagation.

Therefore, knowing $X_0$ and $Y_0$ makes the above equation into a forward scheme for $X_t$ and $Y_t$. We sample $N$ paths of Brownian motion and Poisson process, and employ Euler–Maruyama forward scheme to evaluate the equation \eqref{eq: fbsde} forward in time. At each timestep, we make an ansatz $Z_{i} = \phi_{\theta_i}(X_i, Y_i)$ and $R_{i} = \phi_{\xi_i}(X_i, Y_i)$ for the control processes, where $\phi$ and $\psi$ are neural nets parametrized by $\theta_i$ and $\xi_i$, respectively. Naturally, the terminal condition $Y_T = g(X_T)$ will be violated, which motivates our objective:

$$
\inf _{Y_0,\left\{Z_t\right\}_{0 \leq t \leq T},\left\{R_t\right\}_{0 \leq t \leq T}} \mathbb{E}\left[\left|g\left(X_T^{Y_0, Z, R}\right)-Y_T^{Y_0, Z, R}\right|^2\right]
$$

The overall architecture looks like this:

![Architecture of the FBSDE with jumps](/assets/posts/deep-fbsde/diagram.png)
*Architecture of the deep solver for FBSDE with jumps.<br>Model inputs are given in purple, trainable controls in yellow, and state variables resulting from the Euler scheme given in green.*

## Equation with explicit solution

We set to zero the drift of the forward process, i.e. $b\equiv 0$, and use a simple homogeneous Poisson process as a jump driver of the system. The processes, drivers and coefficients have the following dimensionality:

$$
\newcommand{\R}{\mathbb R}
\begin{gather}
\nonumber W \in \R ^m, N \in \R^\ell\\
\nonumber X\in \R^d, Y\in \R^n, Z\in \R^{n\times m}, R \in \R ^{n\times \ell}\\
\nonumber \gamma \in \R^{d \times \ell}, \sigma \in \R^{d\times m} \\
\end{gather}
$$

Assume that $Y=\varphi(X)$ for $\varphi\in C^2(\R^d, E\subset\R^n)$. Plugging it into \eqref{eq: fbsde}, by Ito's lemma with jumps we necessarily obtain

$$
\begin{aligned}
Z_t &= (D \varphi(X_t))^T \cdot \sigma_t \\
R^{(\cdot k)}_t &= \varphi\left(X_t+\gamma ^{(\cdot k)}_t\right) - \varphi (X_t)
\end{aligned}
$$

Assuming that $D \varphi (X) D \varphi (X)^T$ is never singular and $\varphi ^{-1}\in C^2(E, \mathbb R ^d)$ exists, we have

$$
\begin{aligned}
\sigma_t &= \left(D \varphi (X_t) D \varphi (X_t) ^T \right) ^{-1} D \varphi (X_t) Z =: \beta(X_t) Z_t\\
\gamma^{(\cdot k)}_t &= \varphi ^{-1}\left(\varphi(X_t) + R^{(\cdot k)}\right) - X_t
\end{aligned}
$$

for $\beta(X) = \left(D \varphi(X) D \varphi(X) ^T \right) ^{-1} D \varphi(X)$. Thus we obtain by Ito's lemma (note that $D^2 \varphi$ is a 3-tensor indexed by $i,j,k$ with $k\in 1,\dots, n$)

$$
-f\left(t, X_t^{Y_0, Z, R}, Y_t^{Y_0, Z, R}, Z_t\right) dt = \frac 12 \sum_{i,j=1}^d (\beta(X_t) Z_t(\beta(X_t) Z_t)^T)^{(ij)} (D^2 \varphi(X_t))^{(ij)} dt
$$

Altogether, plugging all in, the FBSDE system \eqref{eq: fbsde} takes the form

$$
\begin{align}
\label{eq: explicit fbsde solution}
dX_t &= \beta(X_t) Z_t dW_t + \left(f^{-1}\left(f(X_t) + R^{(\cdot k)}\right) - X_t\right) dN_t\\
\nonumber dY_t &= \frac 12 \sum_{i,j=1}^d (\beta(X_t) Z_t(\beta(X_t) Z_t)^T)^{(ij)} (D^2 \varphi(X_t))^{(ij)} dt + Z_t dW_t + R_t dN_t
\end{align}
$$

### Example

We want to consider the case $\varphi(X) = g(X) = \exp(aX)$, elementwise, so that it would hold $Y_t = \exp(aX_t)$, and we would have explicit solution for $Y_0$, i.e. $Y_0 = \exp(aX_0)$. To simplify, we assume that state processes $X_t$ and $Y_t$, and both diffusion and jump drivers have equal dimensions, i.e. $d=\ell=m=n$. It holds then

$$
\begin{aligned}
\varphi ^{-1}(X) &= \frac 1a \log(X)\\
D\varphi (X)^{(ij)} &= a\delta_{ij} \exp(aX^{(i)}) = a\operatorname{diag}(\exp(aX))\\
D^2 \varphi (X)^{(ijk)} &= a^2 \delta_{ijk} \exp(aX^{(i)})\\
\beta(X) &= \frac 1a \delta_{ij}\exp(-aX^{(i)}) = \frac 1a \operatorname{diag}(\exp(-aX))
\end{aligned}
$$

Thus, the system \eqref{eq: explicit fbsde solution} can be rewritten as

$$
\begin{align}
\label{eq: fbsde with exp}
dX_t &= \frac 1a \operatorname{diag}(\exp(-aX)) Z_t dW_t + \left(\frac 1a \log\left(\exp(aX_t) + R^{(\cdot k)}\right) - X_t\right) dN_t\\
\nonumber dY_t &= \frac {1}{2} \operatorname{diag} ( \operatorname{diag}(\exp(-aX)) Z_t( \operatorname{diag}(\exp(-aX)) Z_t)^T) \exp(aX_t) dt + Z_t dW_t + R_t dN_t
\end{align}
$$

where the drift of dY is $k$-dimensional vector, and $\operatorname{diag}$ means either extracting diagonal or converting vector into diagonal matrix, depending on the context. As noted before, for this system holds

$$Y_t = \exp(aX_t).$$

## Implementation and training

We test our algorithm with the equation \eqref{eq: fbsde with exp}, for which we know explicit solution. We set $x_0=0$, elementwise, and expect that upon convergence shoudl hold $Y_0 = \exp(a X_0) = \exp(1)$, elementwise.

We make two separate runs of the algorithm, with minimal settings (#1) and with maximum settings (#2), the parameters are summarized below.

| Parameter                         | Experiment #1 | Experiment #2 |
|-----------------------------------|---------------|---------------|
| Number of paths                   | $2^{12}$      | $2^{14}$      |
| Number of timesteps               | 4             | 16            |
| Time horizon                      | 1             | 1             |
| Dimension of $X$ ($d$)            | 2             | 20            |
| Dimension of $Y$ ($n$)            | 2             | 20            |
| Number of diffusion factors ($m$) | 2             | 20            |
| Number of jumps factors ($\ell$)  | 2             | 20            |
| Intensity of the jump process     | 5             | 5             |
| Epochs                            | 20            | 20            |
| Batch size                        | 32            | 32            |

Note that the size of the neural net increases with $d$, $n$, $m$, $\ell$, which is detrimental for the training process. In addition, increasing the number of timesteps increases the depth of the neural net, which also makes it harder to train the neural net and contrasts with the common situation in the numerical analysis, where decrease of the step size of numerical scheme improves convergence properties.

Below we present convergence of the training loss and the trainable initial value parameter $Y_0$ for the Experiment #1.

![Convergence of experiment #1](/assets/posts/deep-fbsde/FBSDE2.png)
*Training loss (left) and initial value of the adjoint process $Y_0$ (right) for the Experiment #1.<br> The reference value $e$ is given in grey. Both components of $Y_0$ converge to the reference value.*

We display convergence of the training loss and initial value for the Experiment #2 below. Note that the Experiment #2 has 4x as many samples, so each epochs is actually 4x longer than in Experiment #1.

![Convergence of experiment #1](/assets/posts/deep-fbsde/FBSDE20.png)
*Training loss (left) and initial value of the adjoint process $Y_0$ (right) for the Experiment #2.<br> The reference value $e$ is given in grey. All 20 components of $Y_0$ converge uniformly to the reference value.*

We conclude that our extension of the deep FBSDE method performs well even in the presence of the jumps.

> The code for this article is available on [github](https://github.com/starovoitovs/deepfbsde).

# References

* Han, Jiequn, and Arnulf Jentzen. "Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations." Communications in mathematics and statistics 5.4 (2017): 349-380.
* Ji, Shaolin, et al. "Three algorithms for solving high-dimensional fully coupled FBSDEs through deep learning." IEEE Intelligent Systems 35.3 (2020): 71-84.
* Øksendal, Bernt, and Agnes Sulem. Stochastic Control of jump diffusions. Springer Berlin Heidelberg, 2005.

