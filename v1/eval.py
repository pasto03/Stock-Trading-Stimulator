"""
model evaluation
"""


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from trading_agent import trading_agent


# evaluation
num_evals = 10
end_funds = []
for i_eval in tqdm(range(num_evals)):
    agent, _env, user_fund_record, user_action_record, reward_record = trading_agent(num_episodes=1)
    # 有可能最后一天的时候用户持仓量多造成用户没有足够资金，所以我们的end_fund应该是：当前资金 + (用户股票数 * 最终股价)
    end_fund = _env.fund + (_env.user_stocks * _env.stock_prices[-1])
    end_funds.append(end_fund)


# stats
START_FUND = 1000
mean_growth = '{:.2f} %'.format(((np.mean(end_funds) - START_FUND) / START_FUND) * 100)   # x % from original fund
median_growth = '{:.2f} %'.format(((np.median(end_funds) - START_FUND) / START_FUND) * 100)
print(f'Mean fund growth: {mean_growth}')
print(f'Median fund growth: {median_growth}')


# visualization
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.hist(end_funds, bins=100)
plt.title('End funds')

plt.subplot(2, 1, 2)
fund_growth_list = [1 if i > START_FUND else 0 for i in end_funds]
growth_count = sum(fund_growth_list)
non_growth_count = len(end_funds) - growth_count
growth_perc = round(growth_count / len(end_funds) * 100, 2)
non_growth_perc = round(non_growth_count / len(end_funds) * 100, 2)
plt.bar([0, 1], [growth_count, non_growth_count])
plt.title(f'Growth ratio: {growth_perc} % growth and {non_growth_perc} % non-growth')

plt.show()
