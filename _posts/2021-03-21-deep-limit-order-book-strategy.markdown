---
layout: post
title:  "Deep order book prediction model"
date:   2021-03-21
categories: algo-trading
---

> The code for this article is available on [github](https://github.com/starovoitovs/deribot/).

We build a deep model for prediction of market moves based on the recent order book history. The model is based on the [DeepLOB](https://arxiv.org/pdf/1808.03668.pdf) paper and consists of the convolutional and recurrent elements. A sequence of convolutional layers enables automatic feature learning, while the recurrent module captures temporal dependence. Portfolio constructed based on the model predictions leads to positive long-term P&L on the testing dataset, modulo transaction fees.

## Data

We pulled ~3h worth of LOB data for `BTC-PERPETUAL` contract across several days from `deribit.com`. We use data from one day for training and validate/backtest using data for another day. Splitting the dataset from a single day and using one half to train and another to validate/backtest yields slightly better results (perhaps there is a presence of a certain regime in the market).

The model takes prices and volumes of 10 bids/asks from the top of the book for the 100 most recent timesteps as the input. The input is thus a 400-dimensional vector.

## Preprocessing and feature engineering

The targets are built based on the difference of future and past moving averages, the "memory" parameter (length of the sliding window) can be chosen based on the sampling rate and market activity/liquidity. We convert the problem into a classification task by quantizing the labels to -1/0/+1 based on the specified threshold hyperparameter. The model thus outputs probabilities of the down-move/no-move/up-move after several ticks. Both prediction horizon and memory parameter are tunable hyperparameters based on the market activity. If the threshold value is too high (i.e. we try to capture only sizable market moves), the classes are going to be imbalanced and the prediction power of the model lower. The threshold value is thus chosen to indicate a move of the magnitude of several dollars.

## Model

The DeepLOB model consists of a convolutional, recurrent and fully-connected modules. The convolutional layer is aimed extract interactions first between price and volume at the same price level, then between bid and ask at the same depth, and finally across all levels on both sides. The intermediate convolutions over time dimensions is aimed at extracting features at different timescales. This allows to extract various imbalance or momentum indicators automatically, instead of handcrafting them manually. The Keras model is illustrated below:

{% highlight python %}

input_layer1 = Input(shape=(window_size, DEPTH * 4, 1))
input_layer2 = Input(shape=(window_size, N_FEATURES))

x = Conv2D(16, kernel_size=(1, 2), strides=(1, 2), activation='relu')(input_layer1)
x = Conv2D(16, kernel_size=(4, 1), activation='relu', padding='same')(x)
x = Conv2D(16, kernel_size=(4, 1), activation='relu', padding='same')(x)

x = Conv2D(16, kernel_size=(1, 2), strides=(1, 2), activation='relu')(x)
x = Conv2D(16, kernel_size=(4, 1), activation='relu', padding='same')(x)
x = Conv2D(16, kernel_size=(4, 1), activation='relu', padding='same')(x)

x = Conv2D(16, kernel_size=(1, DEPTH), activation='relu')(x)
x = Conv2D(16, kernel_size=(4, 1), activation='relu', padding='same')(x)
x = Conv2D(16, kernel_size=(4, 1), activation='relu', padding='same')(x)

x = Reshape((WINDOW_SIZE, 16))(x)

x = concatenate([x, input_layer2])

lstm_layer = LSTM(64)(x)
bn_layer = BatchNormalization()(lstm_layer)
dropout_layer = Dropout(0.8)(bn_layer)
output_layer = Dense(3, activation='softmax')(dropout_layer)

model = Model([input_layer1, input_layer2], output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

{% endhighlight %}


## Training

The training results are given below.

![Training results](/assets/posts/deep-limit-order-book-strategy/training.png)
*Training results (random guessing would give accuracy of ~33.3%)*

It seems that the model is able to capture obvious imbalances and momentum. Higher precision of the neutral signal ensures that we have to enter the coin-flip situation less often.

![Predictions](/assets/posts/deep-limit-order-book-strategy/predictions.png)
*Top: running predictions of the probability of the market move for 1-minute period (softmax outputs: +1 in red, 0 in grey, -1 in blue).<br>Bottom: best bid of BTC-PERPETUAL for the same period. Chosen strategy is colorbar at the bottom.<br>(red/blue/grey for long/short/none, respectively)*

## Portfolio construction model

In the original paper the authors act on the signal by longing/shorting a single futures contract and retain the position until the opposite signal prevails (in order to avoid buying/selling on a neutral signal). However, to avoid redundant market frictions, we have smoothed the strategy with EWMA in order to give up the position only after the neutral signal has been persisting for a longer period of time.

![PNL](/assets/posts/deep-limit-order-book-strategy/pnl.png)
*Top: best bid for BTC-PERPETUAL for 3 hours. Bottom: PNL profile for the same time period without consideration of the fees. Chosen strategy is displayed as colorbar at the bottom (red/blue/grey for long/short/none, respectively).*

## Fees

The P&L graph above is surely way too optimistic, since it assumes we can execute with market orders for free. In order to capitalize on the predictions, one has to cross the spread (if ones places limit order, the market moves in the opposite direction, and the limir order would never get filled).

Lowest fees in the crypto are ~0.05% for liquidity takers, and 0.00% or even a small rebate for liquidity makers. There are some exchanges boasting no fees, but they have huge spreads and ticks. Given the current value of around ~30k for BTC/USD pair, it amounts to <span>$</span>15 per trade. So the model has to predict a market move of ><span>$</span>15 on average. Obviously, the objective is to remove the number of trades and only enter a position if the signal is strong enough to beat the ~<span>$</span>15 fees per contract purchased.

The model is, however, not perfectly accurate, and the predicted jumps are not always that large. In the paper there wasn't a lot of focus on the portfolio construction model, perhaps relying on the assumption that large players have a lot of market power and barely incur fees. One could consider Kelly criterion and size the bets based on the confidence of the model prediction, but it did not yield significant improvement in this case. Alternatively, one could build a market making strategy. For example, using model predictions as short-term alpha indicators in the market making problem by Cartea and Wang.

# References

* Zhang, Zihao, Stefan Zohren, and Stephen Roberts. "Deeplob: Deep convolutional neural networks for limit order books." IEEE Transactions on Signal Processing 67.11 (2019): 3001-3012.
* Cartea, Álvaro, and Yixuan Wang. "Market making with alpha signals." International Journal of Theoretical and Applied Finance 23.03 (2020): 2050016.

