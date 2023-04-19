"""
A method called `pretrained_trading_agent` was defined to pretrain the agent before application
"""

import torch
import matplotlib.pyplot as plt
from stimulator_env import StockStimulatorEnv
from stock_generator import StockGenerator
from model import DQNAgent


# define hyperparameters
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.99
memory_size = 32
batch_size = 32
lr = 0.005


def trading_agent(num_episodes, stock_data=[], day_limit=1000, initial_fund=1000, buy_ratio=0.5, end_log=True):
    # 初始化环境、智能体和记忆池
    # 限制initial_fund不能小于1000
    initial_fund = max(initial_fund, 1000)
    env = StockStimulatorEnv(stock_data=stock_data, initial_fund=initial_fund, day_limit=day_limit, buy_ratio=buy_ratio, random_seed=22)
    state_dim = env.state_dim
    action_dim = env.action_dim
    agent = DQNAgent(state_dim, action_dim, gamma=gamma, lr=lr, epsilon=epsilon_start,
                     epsilon_min=epsilon_end, epsilon_decay=epsilon_decay)
    memory = []

    # 训练DQNAgent
    for i_episode in range(num_episodes):
        # 记录agent环境数据
        user_fund_record = []
        user_asset_record = []
        user_action_record = []
        reward_record = []

        _, state = env.reset()
        done = False
        info = dict()
        while True:
#             try:
            # stopping condition
            if done:
                if end_log:
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
            user_asset_record.append(env.current_asset)
            user_action_record.append(action)
        #             print(env.hold_record)
#             except Exception as e:
#                 print(e)
#                 break
        # 打印当前episode的总奖励和epsilon值
        print(f"Episode {i_episode + 1}: Total Reward = {sum(reward_record):.3f}, Epsilon = {agent.epsilon:.3f}")

    return agent, memory, env, user_fund_record, user_asset_record, user_action_record, reward_record


def pretrained_trading_agent(pretrain_episodes, stock_data=[], initial_fund=1000,
                             pretrain_days=1000, day_limit=1000, buy_ratio=0.3):
    # 预训练
    print('Pretrain process starts...')
    # 定义预训练参数
    initial_fund = max(initial_fund, 1000)

    # 获取pretrained agent
    pretrained_agent, memory, _env, _, _, _, _ = trading_agent(num_episodes=pretrain_episodes, day_limit=pretrain_days,
                                                               initial_fund=initial_fund, buy_ratio=buy_ratio,
                                                               end_log=False)
    print('Pretrain completed.')

    # 实际训练和评估
    print('\nActual trading process starts...')
    # 初始化环境和记忆池
    #     day_limit = day_limit
    num_samples = pretrain_days + day_limit
    if len(stock_data) > 0:
        stock_data = stock_data[:day_limit]
    else:
        #         stock_data = StockGenerator().brownian(num_samples=num_samples)[pretrain_days:]
        stock_data = StockGenerator().brownian(num_samples=num_samples)[:day_limit]

    env = StockStimulatorEnv(stock_data=stock_data, initial_fund=initial_fund,
                             day_limit=day_limit, buy_ratio=buy_ratio, random_seed=22)
    state_dim = env.state_dim
    action_dim = env.action_dim
    agent = pretrained_agent

    memory = []

    # 记录agent环境数据
    user_fund_record = []
    user_asset_record = []
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
            user_asset_record.append(env.current_asset)
            user_action_record.append(action)
        #             print(env.hold_record)
        except Exception as e:
            print(e)
            break
    # 打印总奖励和epsilon值
    print(f"Total Reward = {sum(reward_record):.3f}, Epsilon = {agent.epsilon:.3f}")
    print(f'End reason: {info}')

    return agent, memory, env, user_fund_record, user_asset_record, user_action_record, reward_record


if __name__ == '__main__':
    # agent, memory, _env, user_fund_record, user_asset_record, user_action_record, reward_record = trading_agent(
    #     num_episodes=3)

    agent, memory, _env, user_fund_record, user_asset_record, \
    user_action_record, reward_record = pretrained_trading_agent(pretrain_episodes=3)

    # visualize result
    print('User ending fund:', user_fund_record[-1])
    print('User ending asset:', user_asset_record[-1])

    stock_prices = _env.stock_prices

    plt.figure(figsize=(16, 9))
    plt.subplot(3, 1, 1)
    plt.plot(user_fund_record)
    plt.title('User Fund Record')

    plt.subplot(3, 1, 2)
    plt.plot(user_asset_record)
    plt.title('User Asset Record')

    plt.subplot(3, 1, 3)
    # Define the color map
    color_map = {0: 'red', 1: 'green', 2: 'gray'}

    # Plot the stock price chart
    plt.plot(stock_prices)

    # Color each data point according to the user's actions
    colors = [color_map[action] for action in user_action_record]
    plt.scatter(range(len(stock_prices)), stock_prices, c=colors)
    plt.title('Trade Record')

    plt.show()
