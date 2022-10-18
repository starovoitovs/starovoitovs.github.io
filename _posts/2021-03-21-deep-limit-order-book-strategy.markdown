---
layout: post
title:  "Deep prediction model based on recent limit order book history"
date:   2021-03-21 12:35:11 +0200
categories: algo-trading
---

> The code for this article can be found on [github](https://github.com/starovoitovs/deribot/).

We consider a deep model for prediction of market moves based on the history of the limit order book. The model is based on the [DeepLOB](https://arxiv.org/pdf/1808.03668.pdf) paper and consists of the CNN and LSTM components. A sequence of CNN layers enables representation learning while LSTM module should capture temporal dependence. As input the model takes prices and volumes of 10 bids and asks closest to the mid-price for the 100 most recent timesteps (thus input is a 400-dimensional vector). The regressors are built based on the difference of future and past moving averages, the "memory" parameter (length of the sliding window) can be chosen based on the sampling rate and market activity/liquidity.

We then convert the problem into a classification task by quantizing the labels to -1/0/+1 based on the specified threshold hyperparameter. The model thus outputs probabilities of the down-move/no-move/up-move after several ticks. Both prediction horizon and memory parameter are tunable hyperparameters based on the market activity. If the threshold value is too high (i.e. we try to capture only sizable market moves), the classes are going to be imbalanced and the prediction power of the model lower. The threshold value is thus chosen to indicate a move of the magnitude of several dollars.

![Training results](/assets/posts/deep-limit-order-book-strategy/training.png)
*Training results (random guessing would give accuracy of ~33.3%)*

## Data

The ~3h worth of LOB data for BTC-PERPETUAL across several days was pulled from deribit.com. We use data from one day for training and validate/backtest using data for another day. Splitting the dataset from a single day and using one half to train and another to validate/backtest yields slightly better results (perhaps there is a presence of a certain regime in the market).

![Predictions](/assets/posts/deep-limit-order-book-strategy/predictions.png)
*Top: predictions for the probability of the market move for 1 minute period. <br>Bottom: best bid of BTC-PERPETUAL for the same period. Chosen strategy is colorbar at the bottom.*

## Portfolio construction model

In the original paper they act on the signal by longing/shorting a single futures contract and retain the position until the opposite signal prevails (in order to avoid buying/selling on a neutral signal). One could perhaps incorporate some ideas on Kelly criterion to size the position, however, in the current context it's not entirely necessary.

However, since the model sometimes isn't quick enough to timely predict the opposite move, I have modified the strategy using EWMA to give up the position after a while if the neutral signal has been around for too long.

![PNL](/assets/posts/deep-limit-order-book-strategy/pnl.png)
*Top: best bid for BTC-PERPETUAL for 3 hours. Bottom: PNL profile for the same time period without consideration of the fees. Chosen strategy is displayed as colorbar at the bottom (red/blue/grey for long/short/none, respectively).*

## Fees

Major problem is the fee structure. In order to capitalize on the predictions, one has to cross the spread and execute market orders (since the markets moves against my limit order, and it would never get filled). Lowest fees one can get in the BTC field are ~0.05% for liquidity takers (0.00% or even a small rebate for liquidity makers, there are some exchanges boasting no fees but they have huge spreads and tick size). Given the current value of around ~30k for BTC/USD pair, it amounts to \\$15 per trade. So the model has to predict a market move of >\\$15 on average. Obviously, the objective is to remove the number of trades and only enter a position if the signal is strong enough to beat the ~\\$15 fees per contract purchased.

The model is, however, not perfectly accurate, and the predicted jumps are not always that large. In the paper there wasn't a lot of focus on the portfolio construction model, perhaps relying on the assumption that large players have a lot of market power and barely incur fees.

One way out of it would be build a strategy with limit orders. However, as I can see it, limit orders could be used to capitalize on the excursion (a down-movement followed by an up-movement and vice versa), but not on a single move up or down.

# References

* Zhang, Zihao, Stefan Zohren, and Stephen Roberts. "Deeplob: Deep convolutional neural networks for limit order books." IEEE Transactions on Signal Processing 67.11 (2019): 3001-3012.

