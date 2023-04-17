"""
the trading agent itself is a simplified version of training model
therefore, recording outputs of trading agent will be an eval process
"""

from trading_agent import trading_agent
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# evaluation
end_funds = []
for i in tqdm(range(100)):
    end_fund, _, _, _, _, _ = trading_agent()
    end_funds.append(end_fund)

START_FUND = 1000
mean_growth = '{:.2f} %'.format(((np.mean(end_funds) - START_FUND) / START_FUND) * 100)   # x % from original fund
median_growth = '{:.2f} %'.format(((np.median(end_funds) - START_FUND) / START_FUND) * 100)
print(f'Mean fund growth: {mean_growth}')
print(f'Median fund growth: {median_growth}')


# visualize output
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.hist(end_funds, bins=100)
plt.title('End funds')

plt.subplot(2, 1, 2)
fund_growth_list = [1 if i > START_FUND else 0 for i in end_funds]
growth_count = sum(fund_growth_list)
non_growth_count = len(end_funds) - growth_count
plt.bar([0, 1], [growth_count, non_growth_count])
plt.title(f'Growth ratio: {growth_count} % growth and {non_growth_count} % non-growth')

plt.show()
