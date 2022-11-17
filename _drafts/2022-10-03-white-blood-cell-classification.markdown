---
layout: post
title:  "White blood cell classification with an adversarial algorithm"
date:   2022-12-03
categories: machine-learning
---

Hematologists are medical professionals trained in blood-related disorders. One of their main duties is classification of blood cells: doctors analyze blood smears of their patients and evaluate content of pathological blood cells that might hint at leukemia, anemia and other diseases. In practice this tedious task is more often than not performed manually, but it clearly lends itself to modern image recognition technology. One has to keep in mind though that images coming from different labs vary in sharpness, brightness, contrast, scale and other properties. Therefore, one aims to develop an algorithm that would be agnostic towards such secondary factors and can confidently discriminate cell images regardless of their origin.

## Dataset

We work with the dataset from the [Help A Hematologist Out challenge](https://helmholtz-data-challenges.de/web/challenges/challenge-page/93/overview). We are given two "source" datasets with labelled images of individual blood cells, and a third smaller "target" dataset with unlabelled images.

![Model architecture](/assets/posts/white-blood-cells/datasets.png)
*Blood cell images from the source datasets (left and center) and target dataset (right).*

Thus, we are facing an unsupervised domain adaptation problem, which stands in contrast with the last year's [VisDA challenge](http://ai.bu.edu/visda-2021/) where the labels for the validation set were present. We want to build a model that will simultaneously learn correlations in the source datasets while maintaining the ability to extrapolate onto the target dataset. Additionally, the class distribution in the datasets is uneven and roughly corresponds to the actual proportions of the white blood cells in blood smears. We have to take that into account since the assignment is scored based on macro f1-score, which gives equal weight to each class regardless of its cardinality.

## Model description

In this section we give a detailed description on the adversarial algorithm for unsupervised domain adaptation by Zhang et al., detailed description can be found [here](https://arxiv.org/abs/1904.05801).

Remember that there are no labels for the target domain in the domain adaptation problems. The idea is that the distributions of scoring functions on source and target domains should not differ considerably. Thus, one aims to introduce a measure of distance between the source distribution $P$ and the target distribution $Q$. This distance should be minimized additionally to the classification loss on the source domain.

The generalization bounds obtained in the paper allow us to reduce the problem of minimization of the error rate on the target domain to the minimization of the sum of the empirical margin loss and empirical margin disparity discrepancy (MDD), which we introduce in the following. The **margin** and **margin loss** are defined as:

$$
\begin{aligned}
\rho_f(x, y) &\triangleq \frac{1}{2}\left(f(x, y)-\max _{y^{\prime} \neq y} f\left(x, y^{\prime}\right)\right) \\
\operatorname{err}_D^{(\rho)}(f) &\triangleq \mathbb{E}_{x \sim D} \Phi_{\rho} \circ \rho_f(x, y)
\end{aligned}
$$

for the function

$$
\Phi_\rho(x) \triangleq \begin{cases}0 & \rho \leq x \\ 1-x / \rho & 0 \leq x \leq \rho \\ 1 & x \leq 0\end{cases}.
$$

Thus, the margin loss favors confident predictions. Next we introduce the measure of discrepancy between two distributions in terms of the margin, i.e. for some hypothesis class $\mathcal F$ we define **margin disparity** and **margin disparity discrepancy (MDD)**:

$$
\begin{aligned}
\operatorname{disp}_D^{(\rho)}\left(f^{\prime}, f\right) &\triangleq \mathbb{E}_D \Phi_{\rho} \circ \rho_{f^{\prime}}\left(\cdot, h_f\right) \\
d_{f, \mathcal{F}}^{(\rho)}(P, Q) &\triangleq \sup _{f^{\prime} \in \mathcal{F}}\left(\operatorname{disp}_Q^{(\rho)}\left(f^{\prime}, f\right)-\operatorname{disp}_P^{(\rho)}\left(f^{\prime}, f\right)\right)
\end{aligned}
$$

The generalization bounds obtained in the paper allow us to reduce the problem of minimization of the error rate on the target domain to the minimization of the sum of the empirical margin loss and empirical MDD:

$$
\min _{f \in \mathcal{F}} \operatorname{err}_{\widehat{P}}^{(\rho)}(f)+d_{f, \mathcal{F}}^{(\rho)}(\widehat{P}, \widehat{Q})
$$

Remember that MDD corresponds to the supremum over hypothesis class. Thus, additionally to the actual classifier $f$, we introduce an auxiliary classifier $f'$ which should be interpreted as the maximizer from the definition of the MDD. Thus, by introducing additional feature extractor $\psi$, we can express the optimization problem as a minimax game:

$$
\begin{gathered}
\min _{f, \psi} \operatorname{err}_{\psi(\widehat{P})}^{(\rho)}(f)+\left(\operatorname{disp}_{\psi(\widehat{Q})}^{(\rho)}\left(f^*, f\right)-\operatorname{disp}_{\psi(\widehat{P})}^{(\rho)}\left(f^*, f\right)\right), \\
f^*=\max _{f^{\prime}}\left(\operatorname{disp}_{\psi(\widehat{Q})}^{(\rho)}\left(f^{\prime}, f\right)-\operatorname{disp}_{\psi(\widehat{P})}^{(\rho)}\left(f^{\prime}, f\right)\right) .
\end{gathered}
$$

However, note difficulties of optimizing for the margin loss with stochastic gradient descent, as pointed out by Goodfellow. Thus, for $\sigma$ the softmax, we express the objective in terms of the usual cross-entropy loss $L$ and modified cross-entropy loss $L'$:

$$
\begin{aligned}
L\left(f\left(\psi\left(x^s\right)\right), y^s\right) & \triangleq-\log \left[\sigma_{y^s}\left(f\left(\psi\left(x^s\right)\right)\right)\right] \\
L\left(f^{\prime}\left(\psi\left(x^s\right)\right), f\left(\psi\left(x^s\right)\right)\right) & \triangleq-\log \left[\sigma_{h_f\left(\psi\left(x^s\right)\right)}\left(f^{\prime}\left(\psi\left(x^s\right)\right)\right)\right] \\
L^{\prime}\left(f^{\prime}\left(\psi\left(x^t\right)\right), f\left(\psi\left(x^t\right)\right)\right) & \triangleq \log \left[1-\sigma_{h_f\left(\psi\left(x^t\right)\right)}\left(f^{\prime}\left(\psi\left(x^t\right)\right)\right)\right]
\end{aligned}
$$

Thus, the modified error rate and discrepancy are:

$$
\begin{aligned}
\mathcal{E}(\widehat{P}) &=\mathbb{E}_{\left(x^s, y^s\right) \sim \widehat{P}} L\left(f\left(\psi\left(x^s\right)\right), y^s\right) \\
\mathcal{D}_\gamma(\widehat{P}, \widehat{Q}) &=\mathbb{E}_{x^t \sim \widehat{Q}} L^{\prime}\left(f^{\prime}\left(\psi\left(x^t\right)\right), f\left(\psi\left(x^t\right)\right)\right) -\gamma \mathbb{E}_{x^s \sim \widehat{P}} L\left(f^{\prime}\left(\psi\left(x^s\right)\right), f\left(\psi\left(x^s\right)\right)\right)
\end{aligned}
$$

The new parameter $\gamma \triangleq \exp(\rho)$ is designed to account for the margin $\rho$. Thus, our minimax optimization problem reads:

$$
\begin{gathered}
\min _{f, \psi} \mathcal{E}(\widehat{P})+\eta \mathcal{D}_\gamma(\widehat{P}, \widehat{Q}) \\
\max _{f^{\prime}} \mathcal{D}_\gamma(\widehat{P}, \widehat{Q})
\end{gathered}
$$

## Validation metrics

Also remember that we are in the unsupervised setup, so conventional loss/accuracy criterion on the validation portion of the source dataset for model selection is not entirely appropriate, one should incorporate both classification and discrepancy loss into model selection criterion. We are instead considering entropy and SND.

## Implementation

The optimization algorithm can be implemented as a single adversarial network with the following architecture:

![Model architecture](/assets/posts/white-blood-cells/arch.png)
*Photo by [xyz](google.de).*

We use `resnet18` serving as the feature extractor $\psi$. The adversarial mechanism is implemented by the "gradient reverse layer", which basically inverts the gradients at the backprop. It also features "warm start", which implies that inverted gradient updates coming from the discriminator are ignored at first, and increase at a certain schedule depending on the epoch count. This should allow the backbone model to learn the source dataset first before incorporating the discrepancy loss.

Hyperparameters:

* trade-off $\eta$
* margin $\gamma$
* SND temperature $\beta$

Instead of using out-of-the-box imagenet values for normalization, we calculate mean and std of each channel for each dataset (which differs considerably from the imagenet). This brought a considerable improvement.

Since the images from different datasets differ in their geometric and graphical characteristics, we apply random blur, color jitter and spatial transformations to the samples. We also apply different crops for different datasets. 

To account for imbalanced classes, we also weigh the losses by class cardinality.

We can sample sample-label pairs $(x^s, y^s)$ from the source distribution $\widehat P$ and samples $(x^t)$ from the target distribution $\widehat Q$.

The trade-off parameter $\eta$ allows to modulate the preference between classification and discrepancy loss.

## Results

> baseline/ablation study

# References

* Jiang, Junguang, Bo Fu, and Mingsheng Long. "Transfer-learning-library." (2020).
* Zhang, Yuchen, et al. "Bridging theory and algorithm for domain adaptation." International Conference on Machine Learning. PMLR, 2019.
* Musgrave, Kevin, Serge Belongie, and Ser-Nam Lim. "Benchmarking Validation Methods for Unsupervised Domain Adaptation." arXiv preprint arXiv:2208.07360 (2022).