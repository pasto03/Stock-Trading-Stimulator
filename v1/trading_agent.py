"""
DQN network and DQNAgent used
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from stimulator_env import StockStimulatorEnv
from processing_functions import softmax
from model import DQNAgent


# define hyperparameters
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.99
memory_size = 32
batch_size = 32
lr = 0.005
target_update = 1


def trading_agent(num_episodes, day_limit=1000):
    # 初始化环境、智能体和记忆池
    day_limit = day_limit
    env = StockStimulatorEnv(day_limit=day_limit, random_seed=22)
    state_dim = env.state_dim
    action_dim = env.action_dim
    agent = DQNAgent(state_dim, action_dim, gamma=gamma, lr=lr, epsilon=epsilon_start,
                     epsilon_min=epsilon_end, epsilon_decay=epsilon_decay)
    memory = []

    # 训练DQNAgent
    num_episodes = num_episodes  # 可以理解为num_epochs
    for i_episode in range(num_episodes):
        # 记录agent环境数据
        user_fund_record = []
        user_action_record = []
        reward_record = []

        _, state = env.reset()
        done = False
        info = dict()
        while True:
            try:
                # stopping condition
                if done:
                    print('Environment achieved \'done\'. End reason: {}'.format(info))
                    break

                # 根据当前状态选择动作
                action = agent.act(state)

                # 在环境中执行动作，获取下一个状态、奖励和终止标志
                next_state, reward, done, info = env.step(action)
                # 记录当前reward
                reward_record.append(reward)

                reward = torch.tensor([reward], dtype=torch.float32).to(agent.device)

                # 将经验存储到记忆池中
                memory.append((state, action, reward, next_state, done))

                # 更新状态
                state = next_state

                # 从记忆池中随机抽取批次的经验，更新DQN网络参数
                agent.replay(memory, batch_size)

                # 更新epsilon值
                agent.update_epsilon()

                # 更新agent环境数据
                user_fund_record.append(env.fund)
                user_action_record.append(action)
            #             print(env.hold_record)
            except Exception as e:
                print(e)
                break
        # 打印当前episode的总奖励和epsilon值
        print(f"Episode {i_episode + 1}: Total Reward = {sum(reward_record):.3f}, Epsilon = {agent.epsilon:.3f}")

    return agent, env, user_fund_record, user_action_record, reward_record


if __name__ == '__main__':
    agent, _env, user_fund_record, user_action_record, reward_record = trading_agent(num_episodes=1)

    # 有可能最后一天的时候用户持仓量多造成用户没有足够资金，所以我们的end_fund应该是：当前资金 + (用户股票数 * + 最终股价)
    end_fund = _env.fund + (_env.stock_prices[-1] * _env.user_stocks)
    print('User ending fund: ', end_fund)

    stock_prices = _env.stock_prices

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
    colors = [color_map[action] for action in user_action_record]
    plt.scatter(range(len(stock_prices)), stock_prices, c=colors)
    plt.title('Trade Record')

    plt.show()
