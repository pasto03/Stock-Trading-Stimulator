# Stock-Trading-Stimulator
A baby-level trading bot try to find optimized solution on stock trading

## Available models:
- <a href="https://github.com/pasto03/Stock-Trading-Stimulator#v0">V0</a>
- <a href="https://github.com/pasto03/Stock-Trading-Stimulator#v1">V1</a>

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


# V1
## What is the difference?
- Model V1 is built using the simplest reinforcement learning algorithm: `Deep Q-Network`.

- A simple multilayer perceptron and a simple DQN agent is chosen to build this model.

### Reward mechanism in environment changes as below:
&nbsp;

1. When the user chooses to buy while having insufficient funds, the following punishment will be given:

    $\large reward = -1 \cdot \frac{x_n - hold\_mean}{hold\_mean} \cdot user\_stocks$
    Where the greater the number of stocks the user holds, the greater the punishment.

2. When the user chooses to sell stocks while not having enough, the following punishment will be given:

    $\large reward = -1 \cdot cll_n \cdot \frac{in\_ratio \cdot fund}{x_n}$
    
    $in_ratio$ represents the ratio of funds available to the agent for purchasing stocks, within the range of [0.1, 0.9].
    
    $fund$ represents the amount of funds currently held by the user.

3. When the user chooses to hold stocks, there are two punishment conditions:

    1. If the user has holdings, and the average stock price of the holdings is lower than the current price, the greater the punishment, the lower the average holding price.

        The punishment formula: $\large reward = -1 \cdot \frac{x_n - hold\_mean}{hold\_mean} \cdot user\_stocks$
        
        $hold\_mean$ represents the current average stock price of the holdings.
        
        $user\_stocks$ represents the number of stocks held by the user.

    2. If the user is able to purchase stocks and the current stock price is lower, but the user chooses not to operate, the greater the punishment, the lower the stock price.
    
        The punishment formula: $\large reward = -1 \cdot cll_n \cdot \frac{in\_ratio \cdot fund}{x_n}$

