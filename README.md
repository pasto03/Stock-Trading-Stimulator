# Stock-Trading-Stimulator
a academic-level trading bot try to find optimized solution on stock trading

&nbsp;
# V0
Model V0 is build using simplest algorithm. Only 3 params included: 
`action_probs = np.ones(env.action_dim) / env.action_dim`

where action_dim = 3


For each step(epoch), params in `action_probs` will be updated. The bot tries to find the best action with input of nearest `100` stock prices.

&nbsp;
## Definition of "buy low, sell high"
### Definition 1:

1. Based on the current stock price and the stock price of the previous n days, we determine whether the current stock price is the local minimum of the period. If so, we regard it as a good time to buy the stock.

2. If the user buys the stock at the time of the local minimum, the reward formula is as follows:

    $\large reward = cll_n \cdot in$

    $x$ is the stock price of the day n-1 days ago

    $in$ is the number of shares purchased

    $cll$ is the confidence level of the local minimum, ranging from 0 to 1, and is calculated as $cll = softmin(x)$.

3. Similarly, for the local maximum reward, the formula is as follows:

    $\large reward = clh_n \cdot (x_n - h_{mean}) \cdot out$

    $out$ is the number of shares sold

    $clh$ is the confidence level of the local maximum, ranging from 0 to 1, and is calculated as $clh = softmax(x)$.

    $nh$ is the number of shares held

    $in val$ is the holding value, which is the amount spent on buying these shares

    $h_{mean}$ is the hold_mean, which is the average holding value of the user, and is calculated as $h_{mean} = \frac{nh}{in val}$.

If the selling price is lower than the holding average price, resulting in a loss, the reward will be negative.

&nbsp;
## Outputs
### Example output of trading bot
<img src='v0\image outputs\trading_bot.png' alt='Trading bot output image'>

<i>Trading bot output</i>

&nbsp;

### Example output in evaluation
<img src='v0\image outputs\evaluation_result.png' alt='Evaluation result'>

<i>Evaluation result with 100 evals</i>
