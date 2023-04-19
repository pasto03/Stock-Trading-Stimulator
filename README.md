# Stock-Trading-Stimulator
A baby-level trading bot try to find optimized solution on stock trading

## Available models:
- <a href="https://github.com/pasto03/Stock-Trading-Stimulator#v0">V0</a>
- <a href="https://github.com/pasto03/Stock-Trading-Stimulator#v1">V1</a>
- <a href="https://github.com/pasto03/Stock-Trading-Stimulator#v11">V1.1</a>

&nbsp;
# V0
Model V0 is build using simplest algorithm. Only 3 params included: 
`action_probs = np.ones(env.action_dim) / env.action_dim`

# V1
Model V1 is built using the simplest reinforcement learning algorithm: `Deep Q-Network`.

A simple multilayer perceptron and a simple DQN agent is chosen to build this model.

# V1.1
Environment modified. Reward calculation changed.

<a href="https://github.com/pasto03/Stock-Trading-Stimulator/blob/main/v1.1/changelog(en_ver)">View changelog</a>


## Example of Outupts

<img src="v1.1\image outputs\trading_bot-v1.1.png" alt="trading bot output">

<i>v1.1 trading bot output</i>

&nbsp;

<img src="v1.1\image outputs\grid_search_result-v1.1.png" alt="grid search output">

<i>v1.1 grid search output</i>

&nbsp;

<img src="v1.1\image outputs\evaluation_result-v1.1.png" alt="evaluation output">

<i>v1.1 evaluation output</i>