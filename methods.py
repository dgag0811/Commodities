import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

def stochastic_spread_method(theta: float, mu: float, sigma: float, n: int) -> list:
    """
    :param theta: Speed of reversion
    :param mu: Long-term mean
    :param sigma: Volatility
    :param n: Number of trading days
    :return list: stochastic spread
    Simulating an Ornstein-Uhlenbeck process for the spread between two assets
    """
    np.random.seed(42)
    spread = np.zeros(n)
    spread[0] = 0  # Initial spread
    dt = 1 / n
    for t in range(1, n):
        spread[t] = spread[t - 1] + theta * (mu - spread[t - 1]) * dt + sigma * np.sqrt(dt) * np.random.normal()

    return list(spread)

def combined_forecast_method():
    """
    The Combine Forecast Method involves using multiple forecasting models to predict the spread between two assets and then combining the predictions to generate a trade signal.
    The combined forecast can come from various methods like distance, cointegration, or stochastic models.
    :return:
    """
    np.random.seed(42)
    price_1 = np.random.normal(100, 1, 100)
    price_2 = price_1 + np.random.normal(0, 1, 100)

    # Forecasting models: Distance (simple difference) and Cointegration (using linear regression)
    distance = price_1 - price_2  # Distance forecast
    X = price_1.reshape(-1, 1)  # Feature: Asset 1
    model = LinearRegression().fit(X, price_2)  # Fit linear regression for cointegration
    cointegration_forecast = model.predict(X)  # Cointegration forecast

    # Combine the forecasts (e.g., simple average)
    combined_forecast = (distance + (price_2 - cointegration_forecast)) / 2
    return combined_forecast

# Simulate spread using Ornstein-Uhlenbeck process
spread = stochastic_spread_method(0.7, 0, 0.1, 252)

# Plot the simulated spread
plt.plot(spread, label="Simulated Spread")
plt.axhline(0, color='r', linestyle='--', label="Mean")
plt.title("Stochastic Spread Method (Ornstein-Uhlenbeck)")
plt.legend()
plt.show()


# Example: Simulating prices of two assets
combined_forecast = combined_forecast_method()
# Plot the combined forecast
plt.plot(combined_forecast, label="Combined Forecast")
plt.title("Combine Forecast Method")
plt.legend()
plt.show()
