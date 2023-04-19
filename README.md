# Stock-Trading-Stimulator
A baby-level trading bot try to find optimized solution on stock trading

## Available models:
- <a href="https://github.com/pasto03/Stock-Trading-Stimulator#v0">V0</a>
- <a href="https://github.com/pasto03/Stock-Trading-Stimulator#v1">V1</a>
- <a href="https://github.com/pasto03/Stock-Trading-Stimulator/blob/main/README.md#v11">V1.1</a>

&nbsp;
# V0
Model V0 is build using simplest algorithm. Only 3 params included: 
`action_probs = np.ones(env.action_dim) / env.action_dim`

## Outputs
### Example output of trading bot
<img src='v0\image outputs\trading_bot.png' alt='Trading bot output image'>

<i>Trading bot output</i>

&nbsp;

### Example output in evaluation
<img src='v0\image outputs\evaluation_result.png' alt='Evaluation result'>

<i>Evaluation result with 100 evals</i>


# V1
Model V1 is built using the simplest reinforcement learning algorithm: `Deep Q-Network`.
A simple multilayer perceptron and a simple DQN agent is chosen to build this model.

# V1.1
<a href="https://github.com/pasto03/Stock-Trading-Stimulator/blob/main/v1.1/changelog(en_ver)">View changelog</a>
