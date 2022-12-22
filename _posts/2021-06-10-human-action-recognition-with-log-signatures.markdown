---
layout: post
title:  "Human action recognition with log-signatures"
date:   2021-06-10
categories: machine-learning
---

In this article we look at the signature-based algorithm for skeleton-based human action recognition implemented by our team of colleagues from Berlin and Oxford, as part of the [ICCVW2021 MMVRAC competition](https://sutdcv.github.io/multi-modal-video-reasoning/#/datasets). We implement the [PT-Logsig-RNN model](https://arxiv.org/abs/2110.13008) by Liao et al., which combines extraction of the log-signature with convolutional and recurrent modules to transform the spatio-temporal skeletal data.

## Dataset

Detailed description of the dataset can be found in [UAV-Human: A Large Benchmark for Human Behavior Understanding with
Unmanned Aerial Vehicles](https://arxiv.org/abs/2104.00946). While some datasets for human action recognition have pretty rudimentary labels (standing up, sitting down, walking etc.), human actions in UAV-Human dataset have richer semantic meaning (e.g. cutting trees, reading a book, even stealing).

![Comparison](/assets/posts/log-signatures/dataset.png)
*Examples of action videos used for the dataset. We worked with the skeleton data (second on the top-left).*

The dataset was created from drone footage. For each video frame, 17 major body points were labelled manually, as illustrated below. Since the joints are labelled in 3d space and some actions can have two persons on the scene, each frame of the sample consists of $$17\cdot 3\cdot 2=102$$ data points. In total, there are 16718 samples corresponding to 155 classes, which poses quite a challenging classification task.

![Comparison](/assets/posts/log-signatures/joints.png)
*Samples for pose estimation.*

## Preprocessing

For preprocessing, we consider the following steps:

1. Extend shorter samples by looping them.
2. Center the human at origin.
3. Rotate human to align specified joints to x-axis and z-axis.

Another problem encountered in the dataset is jittering of the actors in the scene, resulting from noisy labelling of the joints. In order to address that, we smoothened samples by applying the `savgol` filter, which removed a fair amount of jittering.

In addition, if there are two actors in the scene, quite often they might be switched in different frames (person #1 becomes person #2 for a short while, and vice versa). And some samples are just so wild that you could barely recognize what's going on. Thus, to address the problem of highly pathological samples, we calculate the *energy* of each sample, which basically measures the Euclidean distance between the frames – higher energy indicating lower confidence score – and augment the data with this score.

## Signatures

Before I present the algorithm in detail, here is a short primer of the topic of signatures. At its core, signature of a multi-dimensional signal is a collection of iterated integrals of its different components. Conceptually, a signature of a signal is somewhat similar to the Fourier series, in the sense that it encodes information from the time domain in the signature domain. We write for the iterated integrals (considered as an element in the tensor algebra):

$$
X^I_J=\int_{\substack{u_1<\cdots<u_k \\ u_1, \ldots, u_k \in J}} d X_{u_1}^{\left(i_1\right)} \otimes \cdots \otimes d X_{u_n}^{\left(i_n\right)}
$$

Then the signature of the given signal is the collection of the iterated integrals:

$$
S(X)_J=\left(1, X_J^1, \ldots, X_J^k, \ldots\right)
$$

The step-$k$ *truncated* signature is the signature including iterated integrals up to degree $k$. Remarkably, if one has the knowledge of the entire signature, one can determine the signal in the time domain, modulo a certain negligible equivalence (*tree-like equivalence*, to be precise). In practice, one considers a truncated signature (for example, truncation up to level 2 might be adequate for some applications). 

Notably, signatures satisfy universal approximation property: any continuous functional on the signature can be approximated arbitrarily well by linear functionals on the signature. In theory, if you compute the signature deep enough, it should suffice to build a linear models on top of it. In practice, however, we only compute the signature up to a certain level, and we work with log-signature which lacks the stated universality property, so throwing a deep model usually would lead to a significant performance improvement.

### Log-signatures

Note that the truncated signature takes values in the associative algebra $T^N\left(\mathbb{R}^d\right):=\oplus_{k=0}^N\left(\mathbb{R}^d\right)^{\otimes k}$, with tensor product service as the product in the algebra. The subset $\mathfrak{t}^N\left(\mathbb{R}^d\right) \equiv \left\\{g \in T^N\left(\mathbb{R}^d\right): \pi_0(g)=0\right\\}$ constitutes a Lie algebra, and one can consider the exponential map from $\mathfrak{t}^N\left(\mathbb{R}^d\right)$ to $1+\mathfrak{t}^N\left(\mathbb{R}^d\right)$, and its inverse – the logarithmic map: 

$$
\begin{aligned}
\exp (a) &= 1+\sum_{k=1}^N \frac{a^{\otimes k}}{k !} \\
\log (1+t)&=\sum_{n=1}^{N} \frac{(-1)^{n-1}}{n} t^{\otimes n} \quad \forall a \in T(E)\\
\end{aligned}
$$

Note that this definition corresponds to the classical power series of $\exp$ and $\log$, truncated up to level $N$, with usual powers replaced with tensor powers. Crucially, logarithmic map is a bijection between $1+\mathfrak{t}^N\left(\mathbb{R}^d\right)$ and $\mathfrak{t}^N\left(\mathbb{R}^d\right)$, and allows a parsimonious representation of the signature features, while being more robust against missing data. For a 102-dimensional path we are dealing with, degree of the 2nd order signature is 10506 and 2nd order degree of the log-signature is 5253. Note that the use of log-signatures instead of vanilla signatures allows us to halve the number of features.

![Comparison](/assets/posts/log-signatures/comparison.png)
*Comparison of the signature and the log-signature.<br>Liao, Shujian, et al. "Learning stochastic differential equations using RNN with log signature features."*

## Model architecture

Instead of calculating log-signatures before the training and feeding them directly to the model, the calculation of the log-signatures is implemented as a Keras layer in the neural net. This allows to prepend path-transformation (PT) layer before the log-signature layer, in order to embed the high-dimensional signal into smaller space with a series of path transformations before the calculation of the log-signatures. 

Implementation of the log-signature as an intermediate layer allows to learn the said embedding by backpropagation. There are various choices of path transformation layers suggested by the authors, depending on the context: linear embedding layer, graph convolutional network, layer augmenting the path with time dimension, and others.

Furthermore, note that we don't just compute a single signature for the entire signal. Instead, we slice the signal into several windows and compute the log-signature for each of them, which is then fed into the LSTM module. Thus we transform the high-frequency signal onto coarser time grid. Varying the length of the window allows capturing dynamics at different time scales. Note that if one would apply the LSTM module directly to the time series, one would necessarily need to downsample the signal, which might lead to the loss of microscopic characteristics. 

The resulting PT-logsig-RNN architecture looks as follows:

![Model architecture](/assets/posts/log-signatures/architecture.png)
*PT-logsig-RNN model architecture.<br>Liao, Shujian, et al. “Logsig-RNN: a novel network for robust and efficient skeleton-based action recognition.”*

## Implementation and training

In summary, the TensorFlow model looks like this:

![TensorFlow model](/assets/posts/log-signatures/model.png)
*TensorFlow model summary.*

The validation loss reaches its minimum fairly quickly, and the validation accuracy plateaus after 10 epochs at ~40%. Given that the total number of classes is 155, we observe quite reasonable performance on the validation set.

![TensorFlow model](/assets/posts/log-signatures/iccv_loss_acc.png)
*Training loss and accuracy.*

> The code for this article is available on [github](https://github.com/RemyMess/MMVRC_ICCV_2021_Skeleton_based_Action_Recognition).

# References

* Li, Tianjiao, et al. "Uav-human: A large benchmark for human behavior understanding with unmanned aerial vehicles." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.
* Liao, Shujian, et al. "Logsig-RNN: a novel network for robust and efficient skeleton-based action recognition." arXiv preprint arXiv:2110.13008 (2021).
* Liao, Shujian, et al. "Learning stochastic differential equations using RNN with log signature features." arXiv preprint arXiv:1908.08286 (2019).

