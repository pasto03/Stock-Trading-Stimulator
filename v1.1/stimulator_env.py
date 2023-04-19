import random
import numpy as np
from stock_generator import StockGenerator
from processing_function import compare_and_count


class StockStimulatorEnv(StockGenerator):
    def __init__(self, stock_data=[], initial_fund=1000, day_limit=1000, buy_ratio=0.5, random_seed=None,
                 generator='brownian'):
        super().__init__()
        if random_seed:
            # 固定生成结果
            np.random.seed(random_seed)

        self.initial_fund = initial_fund
        self.fund = self.initial_fund
        self.user_stocks = 0
        self.day_limit = day_limit
        self.current_day = 0  # in index format
        if len(stock_data) > 0:
            self.stock_prices = stock_data[:day_limit]
        else:
            self.stock_prices = self.brownian(self.day_limit) if generator == 'brownian' else self.gaussian(
                self.day_limit)

        # 记录用户总资产（资金 + 持仓价值）
        self.current_asset = self.initial_fund

        # 定义行为维度和状态维度
        self.action_dim = 3
        self.state_dim = 2

        # 进行reward计算时使用的前n天天数
        self.max_days_before = 100

        # 禁止梭哈 -- 用户不管设置多大都只能到0.9
        self.buy_ratio = min(buy_ratio, 0.9)

        # 记录操作者持仓数据
        self.hold_record = {'num_hold': 0, 'in_value': 0, 'mean_in_value': 0}

    def reset(self):
        self.fund = self.initial_fund
        self.user_stocks = 0
        self.current_asset = self.initial_fund
        self.current_day = 0
        return self.current_day, [self.fund, self.user_stocks]

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

                # 计算当前股价低于百分之多少的最近股票
                cc_low = compare_and_count(current_price, x_prices, by='higher')
                cc_low = 0.005 if cc_low == 0 else cc_low / len(x_prices)

                reward = float(cc_low * buy_amount)

                # 更新持仓数据
                self.hold_record['num_hold'] = self.user_stocks
                self.hold_record['in_value'] += buy_amount * current_price
                self.hold_record['mean_in_value'] = self.hold_record['in_value'] / self.user_stocks

                # 更新用户资产
                self.current_asset = self.fund + (self.hold_record['in_value'])
            else:  # 避免用户在没有足够资金情况下选择购买操作（我们希望用户进行卖出或者持仓）
                #                 hold_mean_value = self.hold_record['mean_in_value']

                #                 # 梭哈比例限制了用户购买，因此会出现用户资金没有到破产却无法购买股票的情况
                #                 if self.user_stocks == 0 or hold_mean_value == 0:
                #                     reward = -0.5
                #                 elif current_price > hold_mean_value:   # 如果当前价格大于均值，那我们希望用户进行抛售操作
                # #                     reward = min(-1 * ((current_price - hold_mean_value) / hold_mean_value) * self.user_stocks, -0.5)
                #                     reward = -1 * ((current_price - hold_mean_value) / hold_mean_value) * self.user_stocks
                #                 else:   # 反之给一个小损失告诉用户应该进行持仓操作
                #                     reward = -0.5
                reward = -1

        elif action == 1:  # sell
            if self.user_stocks > 0:
                # sold random amount of stocks
                sold_amount = random.randint(1, self.user_stocks)
                # update user fund and stock amount
                self.fund += sold_amount * current_price
                self.user_stocks -= sold_amount

                # 更新持仓数据
                hold_mean = self.hold_record['mean_in_value']
                self.hold_record['num_hold'] = self.user_stocks
                self.hold_record['in_value'] -= hold_mean * sold_amount  # 扣除原来持仓的价值
                if self.user_stocks > 0:
                    self.hold_record['mean_in_value'] = self.hold_record['in_value'] / self.user_stocks
                elif self.user_stocks == 0:
                    self.hold_record['in_value'] = 0
                    self.hold_record['mean_in_value'] = 0

                # 更新用户资产
                self.current_asset = self.fund + (self.hold_record['in_value'])

                # 更新reward
                x_prices = self.stock_prices[
                           max(0, self.current_day - self.max_days_before):self.current_day + 1]  # 前n天到当前价格

                # 计算当前股价大于百分之多少的最近股票
                cc_high = compare_and_count(current_price, x_prices, by='lower')
                cc_high = 0.005 if cc_high == 0 else cc_high / len(x_prices)

                #                 hold_mean_value = self.hold_record['mean_in_value']
                # current_price - hold_mean 数字越大reward越高

                reward = float(cc_high * (current_price - hold_mean) * sold_amount)

            else:  # 避免用户在股票不足的情况下选择卖出操作(我们希望用户此时进行购买操作)
                #                 # 用户选择”抛售“时，股票不足的损失计算方法
                #                 operable_fund = self.buy_ratio * self.fund
                #                 # x_prices = self.stock_prices[900:]
                #                 # cl_h = softmax(x_prices)[-1]
                #                 if operable_fund > current_price:   # 用户可操作资金大于当前价格，那我们希望用户选择购买或者持仓
                #                 #     reward = float(-1 * cl_h * (operable_fund / current_price))
                # #                     reward = min(float(-1 * (operable_fund / current_price)), -0.5)
                #                     reward = float(-1 * (operable_fund / current_price))
                #                 else:   # 如果资金也不足，那可能得宣布破产了
                #                     reward = -1
                reward = -1

        elif action == 2:
            reward = 0.5

        self.current_day += 1
        done = self.current_day >= self.day_limit
        info = dict()
        if done:
            info = {
                'end_reason': f'self.current_day=={self.current_day}, which is larger than or equal to self.day_limit=={self.day_limit}'}
        #         return self.current_asset, reward, done, info
        return [self.fund, self.user_stocks], reward, done, info


if __name__ == '__main__':
    env = StockStimulatorEnv()
    action = random.randint(0, 2)
    print('Action:', action)
    print(env.step(action))
    current_day = env.current_day
    print('Current_day:', current_day)
    current_price = env.stock_prices[current_day]
    print(current_price, env.hold_record['mean_in_value'])
