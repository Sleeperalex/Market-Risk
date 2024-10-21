import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv('natixis_stock.csv', delimiter='\t', names=['Date', 'Price'])
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Price'] = data['Price'].str.replace(',', '.').astype(float)
    data['Return'] = data['Price'].pct_change()
    data = data.dropna()
    print(data,"\n")
    print(data.shape,"\n")
    print(data.info(),"\n")
    

    returns = np.array(get_data(data, '2015-01-01', '2016-12-31')['Return'])

    alpha = 0.01
    var = VaR(returns, alpha)
    es = ES(returns, alpha)
    print(f"VaR at {alpha*100}%: {var*100:.5f}% using a kernel")
    print(f"Expected Shortfall at {alpha*100}%: {es*100:.5f}")

    alpha = 0.05
    var = VaR(returns, alpha)
    es = ES(returns, alpha)
    print(f"VaR at {alpha*100}%: {var*100:.5f}% using a kernel")
    print(f"Expected Shortfall at {alpha*100}%: {es*100:.5f}")

    show_distribution(returns)

def show_distribution(returns):
    sns.histplot(returns, bins=50, kde=True, color='blue')
    plt.show()

def VaR(returns, alpha):
    h=1.06*np.std(returns)*np.power(len(returns),-1/5)
    a,b=-1,1
    for i in range(60):
        c=(a+b)/2
        if kernel_cdf(x=c,X=returns,bandwidth=h) > alpha:
            b=c
        else:
            a=c
    return c

def ES(returns, alpha):
    h=1.06*np.std(returns)*np.power(len(returns),-1/5)
    a,b=-1,1
    for _ in range(60):
        c=(a+b)/2
        if kernel_cdf(x=c,X=returns,bandwidth=h) > alpha/2:
            b=c
        else:
            a=c
    return a

def kernel_cdf(x,X,bandwidth):
    sum = 0
    for i in range(len(X)):
        sum += norm.cdf((x - X[i])/bandwidth)
    return (1/len(X)) * sum
    

def get_data(df, start_date, end_date):
    return df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

if __name__ == "__main__":
    main()
    

