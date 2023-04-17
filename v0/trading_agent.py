"""
v0 only implements the most fundamental learning algorithm
function trading_agent() itself is a training model, with num_epochs == day_limit
"""
from stimulator_env import StockStimulatorEnv
import matplotlib.pyplot as plt
import numpy as np
from processing_functions import softmax


def trading_agent(initial_fund=1000, day_limit=1000, buy_ratio=0.5):
    # Instantiate the environment class
    env = StockStimulatorEnv(initial_fund=initial_fund, day_limit=day_limit, buy_ratio=buy_ratio)
    stock_prices = env.stock_prices
    # Initialize action probability distribution
    action_probs = np.ones(env.action_dim) / env.action_dim
    alpha = 1

    user_fund_record = []
    user_op_record = []
    reward_record = []

    done = None
    info = dict()
    for i in range(day_limit):
        if done:
            end_reason = info.get('end_reason', '')
            if end_reason:
                print(info)
            else:
                print('Day limit reached.')
            break

        # Choose a random action (with biased probs)
        current_action = np.random.choice(env.action_dim, p=action_probs)

        current_day, reward, done, info = env.step(current_action)

        # Adjust the probability distribution according to the reward value
        action_probs[current_action] = action_probs[current_action] + alpha * (reward - action_probs[current_action])
        action_probs = softmax(action_probs).reshape(-1)

        # Record the user's current fund
        user_fund_record.append(env.fund)
        user_op_record.append(current_action)
        reward_record.append(reward)

#     print('User ending fund: ', env.fund)
    return env.fund, action_probs, user_fund_record, user_op_record, reward_record, stock_prices


if __name__ == '__main__':
    end_fund, action_probs, user_fund_record, user_op_record, reward_record, stock_prices = trading_agent()

    print('User ending fund: ', end_fund)

    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.plot(user_fund_record)
    plt.title('User Fund Record')

    plt.subplot(2, 1, 2)
    # Define the color map
    color_map = {0: 'red', 1: 'green', 2: 'gray'}

    # Plot the stock price chart
    plt.plot(stock_prices)

    # Color each data point according to the user's actions
    colors = [color_map[action] for action in user_op_record]
    plt.scatter(range(len(stock_prices)), stock_prices, c=colors)
    plt.title('Trade Record')

    plt.show()

