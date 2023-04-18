"""
define generator class, with gaussian and brownian generator
"""

import numpy as np
import matplotlib.pyplot as plt


class StockGenerator:
    @staticmethod
    def gaussian(num_samples, mean=100, std_dev=12):
        """
        Generate stock data from a normal (Gaussian) distribution.
        Args:
        num_samples: Number of samples to generate
        mean: Mean of the distribution
        std_dev: Standard deviation of the distribution

        Returns:
        NumPy array containing the generated data
        """
        # Generate data from normal distribution
        data = np.random.normal(mean, std_dev, num_samples)

        return data

    @staticmethod
    def brownian(num_samples, init_price=100, drift=0.05, volatility=0.2, time_interval=1):
        """
        Generate stock prices using Brownian motion.

        Args:
        num_samples: Number of samples to generate
        init_price: Initial stock price, default value is 100
        drift: Drift of the stock price, default value is 0.05
        volatility: Volatility of the stock price, default value is 0.2
        time_interval: Time interval, default value is 1

        Returns:
        NumPy array containing the generated data
        """
        # Generate time sequence
        t = np.linspace(0, time_interval, num_samples)

        # Generate random terms
        rand = np.random.normal(0, 1, num_samples)

        # Calculate Brownian motion
        price = init_price * np.exp((drift - 0.5 * volatility**2) * t + volatility * np.sqrt(t) * rand)

        return price


if __name__ == '__main__':
    generator = StockGenerator()

    plt.figure(figsize=(15, 6))

    num_samples = 10000
    plt.subplot(2, 1, 1)
    plt.plot(range(num_samples), generator.brownian(num_samples))
    plt.title('Browninan Stimulator')

    plt.subplot(2, 1, 2)
    plt.plot(range(num_samples), generator.gaussian(num_samples))
    plt.title('Gaussian Stimulator')
    plt.show()
