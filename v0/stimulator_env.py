"""
Assumption to user / agent:
- In env.step(...), when 'done', env.step(...) shall not be called again
"""
from stock_generator import StockGenerator
from processing_functions import softmin, softmax
import numpy as np
import random


class StockStimulatorEnv(StockGenerator):
    def __init__(self, initial_fund=1000, day_limit=1000, buy_ratio=0.5, random_seed=None, generator='brownian'):
        super().__init__()
        if random_seed:
            # Fix the generation result
            np.random.seed(random_seed)

        self.fund = initial_fund
        self.user_stocks = 0
        self.day_limit = day_limit
        self.current_day = 0  # in index format
        self.stock_prices = self.brownian(self.day_limit) if generator == 'brownian' else self.gaussian(self.day_limit)

        # Define the action and state dimensions
        self.action_dim = 3
        self.state_dim = 1

        # Number of days used for reward calculation
        self.max_days_before = 100

        # Ban "all in" strategy -- The user can only set it to 0.9 at most
        self.buy_ratio = min(buy_ratio, 0.9)

        # Record the user's holding information
        self.hold_record = {'num_hold': 0, 'in_value': 0, 'mean_in_value': 0}

    def reset(self):
        self.fund = 0
        self.current_day = 0
        return self.fund, self.current_day

    def step(self, action):
        reward = 0
        current_price = self.stock_prices[self.current_day]
        if self.fund < 0 or (self.fund < current_price and self.user_stocks == 0):
            done = True
            return self.current_day, reward, done, {'end_reason': f'bankrupt: fund = {self.fund}'}
        if action == 0:
            if self.fund > current_price:
                # buy random amount of stocks
                min_in = max(1, (int(self.buy_ratio * self.fund) // current_price))
                buy_amount = random.randint(1, min_in)
                # update user fund and stock amount
                self.fund -= buy_amount * current_price
                self.user_stocks += buy_amount

                # calculate reward
                x_prices = self.stock_prices[
                           max(0, self.current_day - self.max_days_before):self.current_day + 1]  # 前n天到当前价格
                cl_low = softmin(x_prices)
                reward = float(cl_low[-1] * buy_amount)

                # update holding information
                self.hold_record['num_hold'] = self.user_stocks
                self.hold_record['in_value'] += buy_amount * current_price
                self.hold_record['mean_in_value'] = self.user_stocks / self.hold_record['in_value']

        elif action == 1:
            if self.user_stocks > 0:
                # sold random amount of stocks
                sold_amount = random.randint(1, self.user_stocks)
                # update user fund and stock amount
                self.fund += sold_amount * current_price
                self.user_stocks -= sold_amount

                # calculate reward
                x_prices = self.stock_prices[
                           max(0, self.current_day - self.max_days_before):self.current_day + 1]  # 前n天到当前价格
                #                 print(x_prices)
                cl_high = softmax(x_prices)

                hold_mean = self.hold_record['mean_in_value']
                reward = float(cl_high[-1] * (current_price - hold_mean) * sold_amount)

                # update holding information
                self.hold_record['num_hold'] = self.user_stocks
                self.hold_record['in_value'] -= hold_mean * sold_amount  # 扣除原来持仓的价值
                self.hold_record['mean_in_value'] = self.user_stocks / self.hold_record['in_value']

        elif action == 2:
            pass

        self.current_day += 1
        done = self.current_day >= self.day_limit or self.fund < 0
        return self.current_day, reward, done, {}


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
