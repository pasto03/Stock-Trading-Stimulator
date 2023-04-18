"""
changes to reward made in v1
"""
from stock_generator import StockGenerator
from processing_functions import softmin, softmax
import numpy as np
import random


class StockStimulatorEnv(StockGenerator):
    def __init__(self, initial_fund=1000, day_limit=1000, buy_ratio=0.5, random_seed=None, generator='brownian'):
        super().__init__()
        if random_seed:
            # 固定生成结果
            np.random.seed(random_seed)

        self.initial_fund = initial_fund
        self.fund = self.initial_fund
        self.user_stocks = 0
        self.day_limit = day_limit
        self.current_day = 0  # in index format
        self.stock_prices = self.brownian(self.day_limit) if generator == 'brownian' else self.gaussian(self.day_limit)

        # 定义行为维度和状态维度
        self.action_dim = 3
        self.state_dim = 1

        # 进行reward计算时使用的前n天天数
        self.max_days_before = 100

        # 禁止梭哈 -- 用户不管设置多大都只能到0.9
        self.buy_ratio = min(buy_ratio, 0.9)

        # 记录操作者持仓数据
        self.hold_record = {'num_hold': 0, 'in_value': 0, 'mean_in_value': 0}

    def reset(self):
        self.fund = self.initial_fund
        self.user_stocks = 0
        self.current_day = 0
        return self.fund, self.current_day

    def step(self, action):
        current_price = self.stock_prices[self.current_day]
        if self.fund < 0 or (self.fund < current_price and self.user_stocks == 0):
            done = True
            reward = 0
            return self.current_day, reward, done, {'end_reason': f'bankrupt: fund = {self.fund}'}
        if action == 0:  # buy
            if self.buy_ratio * self.fund > current_price:
                # buy random amount of stocks
                min_in = max(1, (int(self.buy_ratio * self.fund) // current_price))
                buy_amount = random.randint(1, min_in)
                # update user fund and stock amount
                self.fund -= buy_amount * current_price
                self.user_stocks += buy_amount

                # 根据定义1给出reward
                x_prices = self.stock_prices[
                           max(0, self.current_day - self.max_days_before):self.current_day + 1]  # 前n天到当前价格
                cl_low = softmin(x_prices)
                reward = float(cl_low[-1] * buy_amount)

                # 更新持仓数据
                self.hold_record['num_hold'] = self.user_stocks
                self.hold_record['in_value'] += buy_amount * current_price
                self.hold_record['mean_in_value'] = self.hold_record['in_value'] / self.user_stocks
            else:  # 避免用户在没有足够资金情况下选择购买操作
                x_prices = self.stock_prices[
                           max(0, self.current_day - self.max_days_before):self.current_day + 1]  # 前n天到当前价格
                # 当前价格越高，惩罚越重
                cl_high = softmax(x_prices)
                #                 reward = float(-1 * cl_high[-1] * self.user_stocks  * x_prices[-1])
                reward = float(-1 * cl_high[-1] * self.user_stocks)

        elif action == 1:  # sell
            if self.user_stocks > 0:
                # sold random amount of stocks
                sold_amount = random.randint(1, self.user_stocks)
                # update user fund and stock amount
                self.fund += sold_amount * current_price
                self.user_stocks -= sold_amount

                # 根据定义1给出reward
                x_prices = self.stock_prices[
                           max(0, self.current_day - self.max_days_before):self.current_day + 1]  # 前n天到当前价格
                #                 print(x_prices)
                cl_high = softmax(x_prices)

                hold_mean = self.hold_record['mean_in_value']
                reward = float(cl_high[-1] * (current_price - hold_mean) * sold_amount)

                # 更新持仓数据
                self.hold_record['num_hold'] = self.user_stocks
                self.hold_record['in_value'] -= hold_mean * sold_amount  # 扣除原来持仓的价值
                if self.user_stocks > 0:
                    self.hold_record['mean_in_value'] = self.hold_record['in_value'] / self.user_stocks
                else:
                    self.hold_record['mean_in_value'] = 0
            else:  # 避免用户在股票不足的情况下选择卖出操作
                x_prices = self.stock_prices[
                           max(0, self.current_day - self.max_days_before):self.current_day + 1]  # 前n天到当前价格
                # 当前价格越低，惩罚越重
                cl_low = softmin(x_prices)
                #                 reward = float(-1 * cl_low[-1] * max(self.user_stocks, 1)  * x_prices[-1])   # 使用min避免惩罚变为0
                reward = float(-1 * cl_low[-1] * max(self.user_stocks, 1))

        elif action == 2:
            """
            惩罚条件：
            1. 用户有持仓且持仓平均股价比当前价格低，持仓平均越低惩罚越大
            2. 用户能买入股票，且当前股价偏低用户却不操作，股价越低惩罚越大
            """
            reward = 0

            hold_mean_value = self.hold_record['mean_in_value']
            if self.user_stocks > 0 and hold_mean_value < current_price:
                reward = -1 * ((current_price - hold_mean_value) / hold_mean_value) * self.user_stocks
            if self.buy_ratio * self.fund > current_price:
                x_prices = self.stock_prices[
                           max(0, self.current_day - self.max_days_before):self.current_day + 1]  # 前n天到当前价格
                # 当前价格越低，惩罚越重
                cl_low = softmin(x_prices)
                reward = float(-1 * cl_low[-1] * ((self.buy_ratio * self.fund) / current_price))

        self.current_day += 1
        done = self.current_day >= self.day_limit
        info = dict()
        if done:
            info = {
                'end_reason': f'self.current_day=={self.current_day}, which is larger than or equal to self.day_limit=={self.day_limit}'}
        return self.current_day, reward, done, info


if __name__ == '__main__':
    # generate data and operations for 1000 days
    day_limit = 1000
    env = StockStimulatorEnv(day_limit=day_limit, random_seed=2)

    # initialize action probability distribution
    action_probs = np.ones(env.action_dim) / env.action_dim
    alpha = 1

    action = random.choices([i for i in range(env.action_dim)], action_probs)[0]

    # if 'random_seed' is set then the following will always return same result
    # action = np.random.choice(env.action_dim, p=action_probs)

    print(action)
    print(env.step(action))
