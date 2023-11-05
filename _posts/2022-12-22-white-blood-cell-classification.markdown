---
layout: post
title:  "Domain adaptation with an adversarial algorithm for blood cell classification"
date:   2022-12-24
categories: machine-learning
---

In this article, we'll present a machine learning model for classification of blood cells in hematology. We'll emphasize how this algorithm leverages image recognition technology and, importantly, utilizes domain adaptation techniques to effectively handle variations in images from different labs, resulting in more accurate and practical diagnostics.

> The code for this article is available on [github](https://github.com/starovoitovs/wbc/).

## Problem setup

Hematology is the study of blood, blood-forming tissues, and blood diseases, and accurate diagnosis is critical for the effective treatment of blood disorders. One of the main duties of hematologists is classification of blood cells: doctors analyze blood smears of their patients and evaluate content of pathological blood cells that might hint at leukemia, anemia and other diseases. In practice this tedious task is more often than not performed manually, but it clearly lends itself to modern image recognition technology.

Traditionally, diagnostic and prognostic tools in hematology have been trained on relatively small and homogenous datasets. However, these datasets may not accurately reflect the diversity and complexity of real-world patient populations. This can lead to lower accuracy and less effective treatment decisions. Moreover, images coming from different labs vary in sharpness, brightness, contrast, scale and other properties. Therefore, one aims to develop an algorithm that would be agnostic towards these secondary factors and can confidently discriminate cell images regardless of their origin.

This problem provides a great use case for **domain adaptation**. Domain adaptation techniques can help to improve the generalizability of diagnostic and prognostic tools by allowing them to be trained on larger and more diverse datasets, which reduces the error rate and improves the overall accuracy of these tools.

Additionally, domain adaptation techniques can also be useful for addressing the problem of imbalanced datasets in hematology. Imbalanced datasets, where the number of samples in one class is significantly higher than the number in another class, can cause diagnostic and prognostic tools to be biased towards the majority class. Domain adaptation techniques can help to correct this bias and improve the performance of these tools on imbalanced datasets.

## Dataset

We work with the dataset from the [Help A Hematologist Out challenge](https://helmholtz-data-challenges.de/web/challenges/challenge-page/93/overview). We are given two *source* datasets with **labelled images** of individual blood cells, and a third smaller *target* dataset with **unlabelled images**. The classes are labelled by medical professionals, but even for them sometimes it's hard to discriminate between the classes, since some classes correspond to different stages of the cell's development, which will be apparent later from the confusion matrix.

We are thus facing an unsupervised domain adaptation problem. This stands in contrast with the last year's [VisDA challenge](http://ai.bu.edu/visda-2021/) where the labels for the validation set were present. Therefore, we aim to build a model that will simultaneously learn correlations in the source datasets while maintaining the ability to extrapolate onto the target dataset.

![Datasets](/assets/posts/white-blood-cells/datasets.png)
*Blood cell images from the source datasets (left and center) and target dataset (right).*

We will call the dataset by the name of the first author who published the data. Below you can find a short description and the link to the original papers:

* `Acevedo_20` dataset: the dataset (Acevedo et al., 2020) contains a total of 17,092 images of individual normal cells, acquired using the automatic analyzer CellaVision DM96, in the Core Laboratory at the Hospital Clinic of Barcelona. The images were obtained during the period 2015-2019 from blood smears collected from patients without infections, hematologic or oncologic diseases, and free of any pharmacologic treatment at the moment of their blood extraction. The images are in jpg format and the size is 360x363. All the images were obtained in the color space RGB and were annotated by expert clinical pathologists.
* `Matek_19` dataset: the Munich AML Morphology Dataset (Matek et al., 2019) contains 18,365 expert-labeled single-cell images taken from peripheral blood smears of 100 patients diagnosed with Acute Myeloid Leukemia at Munich University Hospital between 2014 and 2017, as well as 100 patients without signs of hematological malignancy. The images were obtained in the color space RGB and their size is 400x400 pixels.

We aim to achieve a high f1 macro score on a third dataset, called `WBC`.

* `WBC1` dataset (validation set): a small subpart of the WBC dataset. It is **unlabeled** and can be used for training, evaluation and domain adaptation techniques.
* `WBC2` dataset (test set): a second similar subpart of the WBC dataset.

![Class cardinalities](/assets/posts/white-blood-cells/cardinalities.png)
*Class distribution in the source datasets.*

Apparently, the class distribution in the datasets is uneven. It roughly corresponds to the actual proportions of white blood cells in blood smears. We have to take that into account, since we aim to maximize the **macro f1-score**, which gives equal weight to each class regardless of its cardinality.

### Preprocessing

We apply z-score normalization to each channel of each image in all three datasets. Instead of using original imagenet values, we calculate mean and std of each channel for each dataset separately. Additionally, we incorporate multiple transformations of the images, such as random scale, translation, blur and others, to increase the size of the dataset and make the model more robust to image variations. Specifically, we used the [albumentations](https://albumentations.ai/) package, and the following transformations for one of the datasets:

{% highlight python %}

CenterCrop(always_apply=False, p=1.0, height=345, width=345)
RandomCrop(always_apply=False, p=1.0, height=224, width=224)
Blur(always_apply=False, p=0.5, blur_limit=(3, 7))
RandomFog(always_apply=False, p=0.5, fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08)
ColorJitter(always_apply=False, p=0.5, brightness=[0.7, 1.3], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[0.0, 0.0])
Flip(always_apply=False, p=0.5)
Rotate(always_apply=False, p=0.5, limit=(-180, 180) interpolation=1, border_mode=4, value=None, mask_value=None, rotate_method='largest_box', crop_border=False)
RandomScale(always_apply=False, p=0.5, interpolation=1, scale_limit=(-0.19999999999999996, 0.19999999999999996))
Resize(always_apply=False, p=1, height=224, width=224, interpolation=1)
Normalize(always_apply=False, p=1.0, mean=[0.8209, 0.7282, 0.8364], std=[0.1649, 0.2523, 0.0945], max_pixel_value=255.0)

{% endhighlight %}

## Model description

In this section we give a detailed description on the adversarial algorithm for unsupervised domain adaptation by Zhang et al., detailed description can be found [in the paper](https://arxiv.org/abs/1904.05801).

Recall that there are no labels for the target domain in the domain adaptation problems. The idea is that the distributions of scoring functions on source and target domains should not differ considerably. Thus, one aims to introduce a measure of distance between the source distribution $P$ and the target distribution $Q$, and we denote sampling distribution by $\hat P$ and $\hat Q$, respectively. This distance should be minimized additionally to the classification loss on the source domain.

The generalization bounds obtained in the paper allow us to reduce the problem of minimization of the error rate on the target domain to the minimization of the sum of the empirical margin loss and empirical margin disparity discrepancy (MDD), which we introduce in the following. The samples $(x,y)$ are drawn from the common distribution $D$, and we introduce a cut-off function

$$
\Phi_\rho(x) \begin{cases}0 & \rho \leq x \\ 1-x / \rho & 0 \leq x \leq \rho \\ 1 & x \leq 0\end{cases}.
$$

The **margin** and **margin loss** are defined as

$$
\begin{aligned}
\rho_f(x, y) &= \frac{1}{2}\left(f(x, y)-\max _{y^{\prime} \neq y} f\left(x, y^{\prime}\right)\right) \\
\operatorname{err}_D^{(\rho)}(f) &= \mathbb{E}_{D} \left[\Phi_{\rho} \circ \rho_f(x, y)\right]
\end{aligned}
$$

Thus, the margin loss favors confident predictions. Next we introduce the measure of discrepancy between two distributions in terms of the margin. We write for the labeling function

$$
h_f: x \mapsto \underset{y \in \mathcal{Y}}{\arg \max } f(x, y)
$$

Then, for some hypothesis class $\mathcal F$ we define **margin disparity** and **margin disparity discrepancy (MDD)** as

$$
\begin{aligned}
\operatorname{disp}_D^{(\rho)}\left(f^{\prime}, f\right) &= \mathbb{E}_D \left[\Phi_{\rho} \circ \rho_{f^{\prime}}\left(\cdot, h_f\right)\right] \\
d_{f, \mathcal{F}}^{(\rho)}(P, Q) &= \sup _{f^{\prime} \in \mathcal{F}}\left(\operatorname{disp}_Q^{(\rho)}\left(f^{\prime}, f\right)-\operatorname{disp}_P^{(\rho)}\left(f^{\prime}, f\right)\right)
\end{aligned}
$$

The generalization bound obtained in the paper estimates the error rate on the *target* domain by the empirical margin loss on the *source* domain and empirical MDD *between source and target distributions*:

$$
\operatorname{err}_Q(f) \leq \operatorname{err}_{\widehat{P}}^{(\rho)}(f)+d_{f, F}^{(\rho)}(\widehat{P}, \widehat{Q}) + \dots
$$

where the remaining terms on the right-hand side don't depend on $f$. Altogether, this bound leads to the **objective**

$$
\min _{f \in \mathcal{F}} \operatorname{err}_{\widehat{P}}^{(\rho)}(f)+d_{f, \mathcal{F}}^{(\rho)}(\widehat{P}, \widehat{Q})
$$

Remember that MDD corresponds to the supremum over hypothesis class. Thus, additionally to the actual classifier $f$, we introduce an auxiliary classifier $f'$ which should be interpreted as the maximizer from the definition of the MDD. Thus, by introducing an additional feature extractor $\psi$ to balance out the maximizer and minimizer, we can express the optimization problem as a **minimax game**:

$$
\begin{gathered}
\min _{f, \psi} \operatorname{err}_{\psi(\widehat{P})}^{(\rho)}(f)+\left(\operatorname{disp}_{\psi(\widehat{Q})}^{(\rho)}\left(f^*, f\right)-\operatorname{disp}_{\psi(\widehat{P})}^{(\rho)}\left(f^*, f\right)\right), \\
f^*=\max _{f^{\prime}}\left(\operatorname{disp}_{\psi(\widehat{Q})}^{(\rho)}\left(f^{\prime}, f\right)-\operatorname{disp}_{\psi(\widehat{P})}^{(\rho)}\left(f^{\prime}, f\right)\right) .
\end{gathered}
$$

### Modification

Note that there are certain difficulties of optimizing for the margin loss with stochastic gradient descent, as pointed out by Goodfellow et al. in their seminal paper on GANs. Thus, denoting by $\sigma_i$ the $i$-th component of the softmax, we express the objective in terms of the cross-entropy loss $L$ and modified cross-entropy loss $L'$:

$$
\begin{aligned}
L\left(f\left(\psi\left(x^s\right)\right), y^s\right) & \triangleq-\log \left[\sigma_{y^s}\left(f\left(\psi\left(x^s\right)\right)\right)\right] \\
L\left(f^{\prime}\left(\psi\left(x^s\right)\right), f\left(\psi\left(x^s\right)\right)\right) & \triangleq-\log \left[\sigma_{h_f\left(\psi\left(x^s\right)\right)}\left(f^{\prime}\left(\psi\left(x^s\right)\right)\right)\right] \\
L^{\prime}\left(f^{\prime}\left(\psi\left(x^t\right)\right), f\left(\psi\left(x^t\right)\right)\right) & \log \left[1-\sigma_{h_f\left(\psi\left(x^t\right)\right)}\left(f^{\prime}\left(\psi\left(x^t\right)\right)\right)\right]
\end{aligned}
$$

(We are effectively labeling samples from source domain with 0 and samples from target domain with 1.) The **modified error rate** and **modified discrepancy** then read:

$$
\begin{aligned}
\mathcal{E}(\widehat{P}) &=\mathbb{E}_{\left(x^s, y^s\right) \sim \widehat{P}} L\left(f\left(\psi\left(x^s\right)\right), y^s\right) \\
\mathcal{D}_\gamma(\widehat{P}, \widehat{Q}) &=\mathbb{E}_{x^t \sim \widehat{Q}} L^{\prime}\left(f^{\prime}\left(\psi\left(x^t\right)\right), f\left(\psi\left(x^t\right)\right)\right) -\gamma \mathbb{E}_{x^s \sim \widehat{P}} L\left(f^{\prime}\left(\psi\left(x^s\right)\right), f\left(\psi\left(x^s\right)\right)\right)
\end{aligned}
$$

The new parameter $\gamma \exp(\rho)$ is designed to account for the margin $\rho$. Ultimately, our **minimax optimization problem** reads:

$$
\begin{gathered}
\min _{f, \psi} \mathcal{E}(\widehat{P})+\eta \mathcal{D}_\gamma(\widehat{P}, \widehat{Q}) \\
\max _{f^{\prime}} \mathcal{D}_\gamma(\widehat{P}, \widehat{Q})
\end{gathered}
$$

The trade-off parameter $\eta$ allows to modulate the preference between classification and discrepancy loss.

## Implementation

The optimization algorithm can be implemented as a single **adversarial network** with the following architecture:

![Model architecture](/assets/posts/white-blood-cells/arch.png)
*Architecture of the adversarial algorithm.<br>Zhang, Yuchen, et al. "Bridging theory and algorithm for domain adaptation." International Conference on Machine Learning. PMLR, 2019.*

We have `resnet18` serving as the feature extractor $\psi$. Importantly, $f$ is not differentiable with respect to the modified loss parameters, whence the adversarial mechanism is implemented by the **gradient reverse layer** (GRL), which basically inverts the gradients at the backprop. The GRL also features *warm start*, which implies that inverted gradient updates coming from the discriminator are ignored at first, and increase at a certain schedule depending on the epoch count. This should allow the backbone model to learn the source dataset first before incorporating the discrepancy loss.

## Training

### Validation metrics

Since we are in the unsupervised setup, merely using the conventional loss/accuracy criterion on the validation portion of the source dataset for model selection is not sufficient. Instead, we incorporate discrepancy loss into model selection criterion. We consider the two following discrepancy metrics: entropy and soft neighborhood density (SND). **Entropy** measures the confidence of the model:

$$
\mathrm { Entropy }=\frac{1}{N} \sum_{i=1}^N H\left(p_i\right)\\
$$

where the entropy function is given by $H\left(p_i\right)=-\sum_{j} p_{i j} \log p_{i j}$, and $p_i$ is the softmax output of the $i$-th sample. **Soft neighborhood density (SND)** computes entropy of the softmaxed target similarity matrix:

$$
\mathrm{SND}=H\left(\operatorname{softmax}_\tau(\widehat{X})\right)
$$

where $F$ is $L^2$ normalized target feature vectors, $X=F^TF$, $\hat X$ is $X$ with diagonal elements removed, and $\mathrm{softmax}_\tau$ is the softmax function with temperature $\tau$. Softmax temperature allows to (de-)emphasize the most confident predictions. High SND indicates that each feature is close to other features and entails good clustering. However, one should be careful, since trivial model mapping all inputs into a single cluster will also yield high SND score.

### Training setup

We conflate both labelled source datasets `Acevedo_20` and `Matek_19` into one source dataset. Using different datasets regularizes the problem and prevents overfitting on each single dataset. We use the unlabelled `WBC1` dataset for the domain adaptation, and hold out the `WBC2` dataset for evaluation.

As a result of hyperparameter tuning, we settle for the following values of the hyperparameters:

* trade-off: $\eta=1$
* margin: $\gamma=4$
* SND temperature: $\beta=0.005$

To account for imbalanced classes, the classes are weighted by their inverse frequency in the source dataset. The model is trained with a batch size of 32 and stopped early after 50 epochs. The learning rate is set to 0.001 and decays by a factor of 0.1 every 20 epochs. The model is trained on a single Nvidia RTX6000 GPU.

### Results

We see that while the transfer loss is plateaus early, the model keeps on learning to discriminate classes on the source dataset. However, adjusting the trade-off $\eta$ between classification and transfer loss did not yield improved score on the validation dataset.

![Loss and accuracy](/assets/posts/white-blood-cells/loss_acc.png)
*Classification and transfer loss (left) and class accuracy (right) during training.*

![Loss and accuracy](/assets/posts/white-blood-cells/entropy_snd.png)
*Entropy and SND during training.*

The confusion matrix for the source dataset is presented below (remember that labels for the validation and test datasets are not available). We see that the model is pretty much able to discriminate the classes, while the most misclassifications are between two types of neutrophils. It's not unexpected, since the two types of neutrophils are very similar in appearance.

![Confusion matrix](/assets/posts/white-blood-cells/cfmatrix.png)
*Confusion matrix for the source dataset.*

Ultimately, we are able to achieve roughly the same macro f1 score and the testing dataset as on the validation dataset, which we used in training for domain adaptation. The score compares with the top submissions of the challenge.

|          | WBC1                | WBC2                |
|----------|---------------------|---------------------|
| micro f1 | 0.7436              | 0.6778              |
| macro f1 | <mark>0.6635</mark> | <mark>0.6513</mark> |

# References

* Zhang, Yuchen, et al. "Bridging theory and algorithm for domain adaptation." International Conference on Machine Learning. PMLR, 2019.
* Musgrave, Kevin, Serge Belongie, and Ser-Nam Lim. "Benchmarking Validation Methods for Unsupervised Domain Adaptation." arXiv preprint arXiv:2208.07360 (2022).
* Ganin, Yaroslav, and Victor Lempitsky. "Unsupervised domain adaptation by backpropagation." International conference on machine learning. PMLR, 2015.
* Jiang, Junguang, Bo Fu, and Mingsheng Long. "Transfer-learning-library." (2020).
