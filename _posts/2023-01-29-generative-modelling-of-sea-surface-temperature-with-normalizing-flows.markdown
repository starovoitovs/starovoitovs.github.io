---
layout: post
title:  "Generative modelling of sea surface temperature with normalizing flows"
date:   2023-02-11
categories: machine-learning
---

In this post we look at the deep flow-based generative model of the evolution of ocean temperature, based on real data provided by Mercator Ocean. We aim to learn the distribution of sea surface temperature (SST) data, including multidimensional dependencies, and generate potential future values of SST with spatial dependency between stations, in particular to simulate extreme climate scenarios within the context of stress testing and, more broadly, climate risk management.

This model was part of the $1^{\text{st}}$ place submission in the [GenHack2](https://www.polytechnique.edu/en/education/academic-and-research-departments/applied-mathematics-department-depmap/student-event/genhack-2-hackathon-generative-modelling) data challenge we took part together with colleagues from HU Berlin, co-organized [Chair Stress test, Risk management and Financial steering](http://www.cmap.polytechnique.fr/~stresstest/) (Ecole polytechnique, BNP Paribas) and [Mercator Ocean](https://www.mercator-ocean.eu/en/).

> The code for this article is available on [github](https://github.com/starovoitovs/sst-gen).

![SST map](/assets/posts/generative-modelling-sst/sstmap.png)
*Map of sea surface temperatures (SST).<br>Image source: ESA SST CCI and C3S reprocessed [sea surface temperature analyses](https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_REP_OBSERVATIONS_010_024/description).*

## Introduction

Climate change poses a significant threat to our planet, with rising temperatures, sea levels, and extreme weather events all having devastating effects on ecosystems and
communities. One important aspect of monitoring and understanding climate change is measuring **sea surface temperature (SST)** in the oceans. This data helps scientists understand
how ocean currents and circulation patterns are changing, which in turn can provide important insights into the overall health of the planet and the potential impacts of climate
change.

Moreover, abnormal variations in SST are a significant threat to marine wildlife, as rising sea surface temperatures can affect the survival and distribution of many species.
Warmer ocean temperatures can lead to coral bleaching, which can kill off entire coral reefs and the diverse marine life that depend on them. It can also disrupt the timing of
reproduction and migration patterns, leading to declines in populations of fish, seabirds, and other species. As such, it is essential to monitor and forecast sea surface
temperature in the oceans in order to better understand and respond to the challenges of climate change. By forecasting temperature trends over time, scientists can identify areas
where species are most at risk and take action to protect them. For example, conservation efforts such as marine protected areas or fishing restrictions in areas where sea surface
temperatures are rising rapidly can help protect vulnerable species.

![Climate change risks](/assets/posts/generative-modelling-sst/risks.png)
*Impacts and risks to ocean ecosystems from climate change.<br>Image source: The Ocean and Cryosphere in a Changing Climate, IPCC 2019.*

According to the [Assessment Report](https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-2/) by the Intergovernmental Panel on Climate Change (IPCC), each of the last four decades has been warmer than any decade that it preceded since 1850. By 2019, the global warming has reached 1.1°C.

![SST history](/assets/posts/generative-modelling-sst/sst_history.png)
*Global surface temperature relative to 1850–1900 (°C).<br>Image source: IPCC [6th Assessment Report](https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-2/), 2019.*

## Problem setup

We are given daily measurements at 6 stations at unknown locations from 1981-09-01 to 2007-12-31. The goal is to generate the distribution of SST at those 6 stations for the next 9 years, one sample a day, from 2008-01-01 to 2016-12-31. The model takes the Gaussian noise $\mathbf z$ as the input, and outputs the vector $\mathbf x$, corresponding to SST at 6 stations on some day in the above interval:

$$
f(\mathbf z)=\mathbf x.
$$

This contrasts with e.g. classical autoregressive time series models, where forecasts are short-term and deterministic. As is common for generative models, the model is evaluated according to the distance between true and modelled sampling distributions, so the order in which the samples are generated does not matter.

Seasonality is removed from the data. The data is provided in the form of a CSV file, where each row corresponds to a single day and each column corresponds to a single station. The SST measurements are usually normalized with respect to some benchmark (e.g. relative to pre-industrial levels). The first column is the date, and the remaining columns are the SST measurements at each station:

| date       | s1     | s2      | s3     | s4     | s5     | s6     |
|------------|--------|---------|--------|--------|--------|--------|
| 1981-09-01 | 1.6863 | -0.7941 | 1.1484 | 1.1581 | 2.0244 | 1.0381 |
| 1981-09-02 | 1.7499 | -0.7423 | 1.0163 | 1.2818 | 2.149  | 0.6016 |
| 1981-09-03 | 1.4752 | -0.7492 | 0.9365 | 1.3864 | 2.1762 | 3.6465 |
| 1981-09-04 | 1.6222 | -0.7246 | 0.939  | 1.5021 | 2.3459 | 3.183  |
| 1981-09-05 | 1.7509 | -0.8586 | 0.9936 | 1.8088 | 2.3881 | 3.0409 |

By looking at yearly means, we observe that the SST do indeed exhibit a positive trend, at all 6 stations:

![Trend](/assets/posts/generative-modelling-sst/trend.png)
*Trend of the sea surface temperatures across 6 stations.*

The histograms of 1D and 2D marginals of the density we are looking to model are depicted below.

![True histogram](/assets/posts/generative-modelling-sst/hist2d_test_true.png)
*Histograms of true 1D and 2D marginal densities for the test set.*

The table below summarizing temperature correlations between different stations indicates that temperatures between some stations is more correlated than between some others (perhaps due to their geographical proximity).

![Correlation matrix](/assets/posts/generative-modelling-sst/corr.png)
*Correlation matrix of the SST observations across the stations.*

## Model

### Normalizing flows

We build a generative model for sea surface temperature with **normalizing flows**. Despite not enjoying the same popularity as other cutting-edge generative models, the complex shape of the distribution and low-dimensional setup make the normalizing flows a great candidate to attack the problem. Normalizing flows models are particularly useful when the data distribution is highly non-Gaussian, has multiple modes, or has complex correlations between the variables.

Normalizing flows model the data distribution by transforming a simple base distribution, such as a Gaussian, into a more complex one. Thereby, one can easily sample from and evaluate density of the simple distribution, and then use composition of simple learnable transformations to obtain the sample from and density of the complex distribution. Thus, unlike in some other generational models, such as GAN or VAE, the transformations in the normalized flows are chosen so that the likelihood can be computed analytically, so the training proceeds by log-likelihood maximization. These simple transformations will be parametrized in our case by neural nets (hence *deep* flow-based model), which allows implementation of the entire model as a neural net and training by backpropagation.

Specifically, let us denote $z$ the variable in the latent space, and $x$ the variable in the data space, and we consider a bijective invertible transformation $f$, which maps one onto the other:

$$
\mathbf z \sim \pi(\mathbf z), \qquad \mathbf x \sim p(\mathbf x), \qquad \mathbf x:=f(\mathbf z), \qquad \mathbf z = f^{-1}(\mathbf x).
$$

Invoking the multivariate **change of variable** formula, we obtain for the density

$$
p(\mathbf{x})=\pi(\mathbf{z})\left|\operatorname{det} \frac{d \mathbf{z}}{d \mathbf{x}}\right|=\pi\left(f^{-1}(\mathbf{x})\right)\left|\operatorname{det} \frac{d f^{-1}}{d \mathbf{x}}\right|.
$$

A common choice for $\pi$ is standard Gaussian, as it also is in our case. Making use of the **inverse function theorem** (Jacobian of the inverse is inverse of the Jacobian), it holds for the Jacobian of the inverse:

$$
\left|\operatorname{det} \frac{d f^{-1}}{d \mathbf{x}}\right| = \left|\operatorname{det} \frac{d f}{d \mathbf{z}}\right|^{-1}
$$

### Autoregressive normalizing flows

The **autoregressive** paradigm in density estimation of sequential data assumes that we model conditional distribution of the next element in the sequence dependent of the previous elements. By the chain rule of probability, the joint density can be factorized into product of one-dimensional conditionals:

$$
\begin{equation}
\label{eq: density estimation}
p(\mathbf{x})=\prod_{i=1}^D p\left(x_i \mid  \mathbf  x_{1:i-1}\right),
\end{equation}
$$

where $D$ is dimension of the output vector. Now we model $f(\mathbf z) = \mathbf x$ with an autoregressive model, where conditionals are one-dimensional Gaussians:

$$
p\left(x_i \mid \mathbf{x}_{1: i-1}\right)=\mathcal{N}\left(x_i \mid \mu_i,\left(\exp \alpha_i\right)^2\right)
$$

where location and scale parameters are functions of the observed part of $\mathbf x$:

$$
\mu_i=g_{\mu_i}\left(\mathbf{x}_{1: i-1}\right) \quad \text { and } \quad \alpha_i=g_{\alpha_i}\left(\mathbf{x}_{1: i-1}\right)
$$

The model $f$ is then an elementwise affine transformation:

$$
x_i=z_i \exp \alpha_i+\mu_i
$$

This autoregressive architecture of the transformation ensures that the Jacobian is triangular, leading to particularly simple form for the modulus of the determinant of the Jacobian of the inverse of the transformation $f$:

$$
\begin{equation}
\label{eq: absdet of the triangular jacobian}
\left|\operatorname{det} \frac{d f^{-1}}{d \mathbf{x}}\right|=\left|\operatorname{det} \frac{d f}{d \mathbf{z}}\right|^{-1}=\exp \left(-\sum_i \alpha_i\right)
\end{equation}
$$

### Masked autoregressive flow

**Masked autoregressive flow (MAF)** is an effective implementation of the autoregressive flow paradigm. It makes use of the **masked autoencoder for distribution estimation (MADE)**, which is a feedforward network allowing computation of the autoregressive parameters $\mu_i$ and $\alpha_i$ in a single forward pass. The autoregressive property is achieved by applying binary mask to the weights of the feedforward net:

![MADE](/assets/posts/generative-modelling-sst/made.png)
*MADE illustration with a simple three-layer MLP.<br>Germain, Mathieu, et al. “Made: Masked autoencoder for distribution estimation.” International conference on machine learning. PMLR, 2015.*

In the case of MADE, the binary mask matrices are constructed based on the ordering of the hidden units, so that the element of the mask matrix equals 1 if the order of the target hidden unit is larger or equal to the order of the source hidden unit, otherwise it's 0.

### Stacking models

Now, instead of considering a single transformation $f$, we stack multiple autoregressive models $f_i$ into a deeper flow:

$$
\mathbf{x}:=\mathbf{z}_K=f_K \circ f_{K-1} \circ \cdots \circ f_1\left(\mathbf{z}_0\right)
$$

We have thus

$$
\mathbf z_{i-1} \sim p_{i-1}(\mathbf z_{i-1}), \qquad \mathbf z_i := f_i(\mathbf z_{i-1}), \qquad \mathbf z_{i-1} = f_i^{-1}(\mathbf z_{i})
$$

Invoking again the change of variable formula, the inverse function theorem and formula for the determinant of the inverse, we obtain: 

$$
p_i\left(\mathbf{z}_i\right) = p_{i-1}\left(f_i^{-1}\left(\mathbf{z}_i\right)\right)\left|\operatorname{det} \frac{d f_i^{-1}}{d \mathbf{z}_i}\right| = p_{i-1}\left(\mathbf{z}_{i-1}\right)\left|\operatorname{det} \frac{d f_i}{d \mathbf{z}_{i-1}}\right|^{-1}.
$$

Iterating the product over $i$, denoting by $\mu_{ij}$ and $\alpha_{ij}$ the location and log-scale parameters of the $i$-th model and $j$-th component, using the Jacobian determinant formula for the autoregressive model $\eqref{eq: absdet of the triangular jacobian}$, and passing to the log-likehood, we obtain:

$$
\log p(\mathbf{x})=\log \pi_0\left(\mathbf{z}_0\right)-\sum_{i=1}^K \log \left|\operatorname{det} \frac{d f_i}{d \mathbf{z}_{i-1}}\right| = \log \pi_0\left(\mathbf{z}_0\right)-\sum_{i=1}^K \sum_{j=1}^D \alpha_{ij}.
$$

Choosing for $\pi_0$ the standard Gaussian, as suggested before, we obtain a tractable formula for the log-likelihood.

### Ad-hoc adjustments

The trend evident from the figure in the Problem Setup section leads us to additionally include a *trend model*, which is a linear model in time and supposed to account for overall increase of the ocean temperature. Additionally, we include a *sample weights model*, which might favor more recent samples as more predictive of the future temperatures.   

## Evaluation

The modelled distribution is going to be compared with the real data to evaluate how well the model can capture the underlying distribution of SST data. The model is scored based on two metrics: distance between 1-dimensional marginals in terms of the **Anderson-Darling distance**, and **absolute Kendall error** which captures the dependency between different stations.

For a vector $\mathbf \xi$, we denote by $\xi_{i,n}$ the order statistic, and $n_{\text {test}}$ is the total number of samples in the test set. We denote by ${\mathbf x}$ the samples of the modelled distribution and by $\widetilde {\mathbf x}$ the samples of the real data.

### One-dimensional marginals: Anderson-Darling distance

We denote by $\widetilde{u}\_{i, n\_{\text {test }}}^s$ the model probability of a generated variable $f(\mathbf z)=\mathbf x$ for a specific station $s$:

$$
\widetilde{u}_{i, n_{\text {test }}}^s=\frac{1}{n_{\text {test }}+2}\left(\sum_{j=1}^{n_{\text {test }}} 1\left\{\widetilde x_j^s \leq x_{i, n_{\text {test }}}^s\right\}+1\right)
$$

The Anderson-Darling distance for each station is computed as

$$
W_{n_{\text {test }}}^s(Z)=-n_{\text {test }}-\frac{1}{n_{\text {test }}} \sum_{i=1}^{n_{\text {test }}}(2 i-1)\left(\log \left(\tilde{u}_{i, n_{\text {test }}}^s\right)+\log
\left(1-\tilde{u}_{n_{\text {test }}-i+1, n_{\text {test }}}^s\right)\right)
$$

The global metric is computed as the average of the 6 station metrics:

$$
\mathcal{L}_M(Z)=\frac{1}{d} \sum_{s=1}^d W_{n_{\text {test }}}^s
$$

### Dependency metric: Absolute Kendall error

Kendall's dependence function captures dependence structure between the features. The estimation of the Kendall’s dependence function is based on the **sampled**
pseudo-observations

$$
R_i=\frac{1}{n_{\text {test }}-1} \sum_{j \neq i}^{n_{\text {test }}} 1\left\{x_j^1<x_i^1, \ldots, x_j^d<x_i^d\right\}
$$

and the **testing** pseudo-observations

$$
\widetilde{R}_i=\frac{1}{n_{\text {test }}-1} \sum_{j \neq i}^{n_{\text {test }}} 1\left\{\tilde{x}_j^1<\tilde{x}_i^1, \ldots, \widetilde{x}_j^d<\widetilde{x}_i^d\right\}.
$$

Then, the **absolute Kendall error** is then given as the $L^1$ distance between those vectors:

$$
\mathcal{L}_D(Z)=\frac{1}{n_{\text {test }}} \sum_{i=1}^{n_{\text {test }}}\left|R_{i, n_{\text {test }}}-\widetilde{R}_{i, n_{\text {test }}}\right|.
$$

## Training

We implement the model in `pytorch` and use the [nflows](https://pypi.org/project/nflows/) library, which contains the implementation of the masked autoregressive flow. We also use `ray` and `hyperopt` for hyperparameter tuning, and `mlflow` for model tracking.

We subdivide the training set into training, validation and test splits. We use Adam optimizer with learning rate 0.0001, and select the following hyperparameters for the MAF:

| Parameter                              | Value |
|----------------------------------------|-------|
| # of stacked layers in MAF             | 4     |
| # of blocks in MADE                    | 5     |
| # of hidden units in MADE inner layers | 8     |

The validation metrics extremize quite early in the training, and attain minimum of 16.16 for Anderson–Darling distance after 39 epochs, and 0.008 for absolute Kendall error after 7 epochs:

![Sampled histogram](/assets/posts/generative-modelling-sst/val_ad_mean.png)
*Anderson–Darling distance on the validation set during training.*

![Sampled histogram](/assets/posts/generative-modelling-sst/val_kendall.png)
*Absolute Kendall error on the validation set during training.*

The sample weights model ended up giving preference to more recent samples in the dataset, and the trend model incorporated a positive trend for SST at all stations, in line with our expectations. We now take a look at the 1D and 2D marginals of the modelled sampling distribution:

![Sampled histogram](/assets/posts/generative-modelling-sst/hist2d_test_ba_pred.png)
*Histograms of the 1D (in orange) and 2D marginals sampled from the modelled distribution,<br>with additional depiction of the 1D marginals on the test set (in blue).*

If we compare with the marginals of the data distribution on the figure at the beginning of the article, we can see that the normalizing flow has been able to accurately represent the complex non-linearities of the density, while accurately capturing both 1D marginals and dependency between different stations.

# References

* Germain, Mathieu, et al. "Made: Masked autoencoder for distribution estimation." International conference on machine learning. PMLR, 2015.
* Papamakarios, George, Theo Pavlakou, and Iain Murray. "Masked autoregressive flow for density estimation." Advances in neural information processing systems 30 (2017).
