"""
A method called `agent_grid_search` was defined to attempt to find the optimal num_episodes
"""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from trading_agent import trading_agent, pretrained_trading_agent


# model evaluation
# evaluation
num_evals = 5
end_funds = []
end_assets = []
for i_eval in tqdm(range(num_evals)):
    agent, memory, env, user_fund_record, user_asset_record, user_action_record, reward_record = trading_agent(num_episodes=3)
    end_funds.append(user_fund_record[-1])
    end_assets.append(user_asset_record[-1])


# stats
START_ASSET = 1000
mean_growth = '{:.2f} %'.format(((np.mean(end_assets) - START_ASSET) / START_ASSET) * 100)   # x % from original fund
median_growth = '{:.2f} %'.format(((np.median(end_assets) - START_ASSET) / START_ASSET) * 100)
print(f'Mean asset growth: {mean_growth}')
print(f'Median asset growth: {median_growth}')


# visualization
plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
plt.hist(end_funds, bins=100)
plt.title('End funds')

plt.subplot(3, 1, 2)
plt.hist(end_assets, bins=100)
plt.title('End assets')

plt.subplot(3, 1, 3)
asset_growth_list = [1 if i > START_ASSET else 0 for i in end_assets]
growth_count = sum(asset_growth_list)
non_growth_count = len(end_assets) - growth_count
growth_perc = round(growth_count / len(end_assets) * 100, 2)
non_growth_perc = round(non_growth_count / len(end_assets) * 100, 2)
plt.bar([0, 1], [growth_count, non_growth_count])
plt.title(f'Growth ratio: {growth_perc} % growth and {non_growth_perc} % non-growth')

plt.show()
