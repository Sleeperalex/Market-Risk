import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

def exercice(data : pd.DataFrame):

    # Set the decay factor
    lambda_ = 0.94
    alpha = 1 - lambda_

    # Calculate exponentially weighted mean and variance
    data['EWM_Mean'] = data['Return'].ewm(alpha=alpha).mean()
    data['EWM_Var'] = data['Return'].ewm(alpha=alpha).var()

    # Latest estimates for mu and sigma
    mu = data['EWM_Mean'].iloc[-1]
    sigma2 = data['EWM_Var'].iloc[-1]
    sigma = np.sqrt(sigma2)

    print(f"Estimated μ (mean return): {mu:.6f}")
    print(f"Estimated σ (volatility): {sigma:.6f}")

    # Number of simulations
    N = 1000

    # Set random seed
    np.random.seed(42)

    # Current stock price
    S0 = data['Price'].iloc[-1]

    # Simulate returns and stock prices
    r_sim = np.random.normal(mu, sigma, N)
    S_T = S0 * np.exp(r_sim)

    # Annualized volatility
    sigma_ann = sigma * np.sqrt(252)

    # Option parameters
    K = S0
    T = 1 / 12
    r = 0


    # Initial call price
    C0 = black_scholes_call_price(S0, K, T, r, sigma_ann)

    # Adjust time to maturity
    T1 = T - (1 / 252)
    T1 = max(T1, 1e-6)

    # Call option prices for simulated stock prices
    C_T = black_scholes_call_price(S_T, K, T1, r, sigma_ann)

    # Change in call prices
    delta_C = C_T - C0

    # Compute VaR
    confidence_level = 0.95
    VaR = -np.percentile(delta_C, (1 - confidence_level) * 100)

    print(f"The VaR at a 95% confidence level is: {VaR:.2f} EUR")

    # Plot distribution
    plt.hist(delta_C, bins=50, edgecolor='black')
    plt.title('Distribution of Changes in Call Option Prices')
    plt.xlabel('Change in Call Price (EUR)')
    plt.ylabel('Frequency')
    plt.axvline(-VaR, color='red', linestyle='dashed', linewidth=2)
    plt.text(-VaR, plt.ylim()[1]*0.9, f'VaR = {-VaR:.2f}', color='red', ha='right')
    plt.show()

def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def main():
    data = pd.read_csv('natixis_stock.csv', delimiter='\t', names=['Date', 'Price'])
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Price'] = data['Price'].str.replace(',', '.').astype(float)
    data['Return'] = data['Price'].pct_change()
    data = data.dropna()
    print(data,"\n")
    print(data.shape,"\n")
    print(data.info(),"\n")

    exercice(data)

if __name__ == "__main__":
    main()