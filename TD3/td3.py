import pandas as pd
import numpy as np

def pickands_estimator(df: pd.DataFrame,column: str, k_n: int):

    data_sorted = df[column].sort_values(ascending=True).values
    
    n=len(data_sorted)

    x1 = data_sorted[n-k_n]
    x2 = data_sorted[n-2*k_n]
    x4 = data_sorted[n-4*k_n]

    xi_pickands = np.log((x1 - x2) / (x2 - x4)) / np.log(2)
    
    return xi_pickands

def hill_estimator(df: pd.DataFrame,column: str, k_n: int):

    data_sorted = df[column].sort_values(ascending=False).values
    n=len(data_sorted)

    xi_hill=0
    for i in range(n-k_n+1,n):
        xi_hill+=np.log(data_sorted[i]/data_sorted[n-k_n+1])

    return xi_hill

def main():
    data = pd.read_csv('natixis_stock.csv', delimiter='\t', names=['Date', 'Price'])
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Price'] = data['Price'].str.replace(',', '.').astype(float)
    data['Return'] = data['Price'].pct_change()
    data = data.dropna()
    print(data,"\n")
    print(data.shape,"\n")
    print(data.info(),"\n")

    data['Return'] = abs(data['Return'])

    k = 6
    xi_estimate = pickands_estimator(data,'Return', k)
    print(f"Pickands' estimate of ξ: {xi_estimate:.4f}")

    xi_estimate = hill_estimator(data,'Return', k)
    print(f"Hill's estimate of ξ: {xi_estimate:.4f}")

if __name__ == "__main__":
    main()