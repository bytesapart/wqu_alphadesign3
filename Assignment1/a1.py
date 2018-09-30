import warnings
from pylab import plot, show
from numpy import vstack,array
from numpy.random import rand
import numpy as np
from scipy.cluster.vq import kmeans,vq
import pandas as pd
from math import sqrt
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from pandas_datareader import data as pdr  # The pandas Data Module used for fetching data from a Data Source
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fix_yahoo_finance import pdr_override  # For overriding Pandas DataFrame Reader not connecting to YF


def yahoo_finance_bridge():
    """
    This function fixes problems w.r.t. fetching data from Yahoo Finance
    :return: None
    """
    pdr_override()


if __name__ == '__main__':
    # Create the Yahoo Finance Bridge so that you get the data
    yahoo_finance_bridge()
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    #read in the url and scrape ticker data
    data_table = pd.read_html(sp500_url)

    tickers = data_table[0][1:][0].tolist()[:100]
    prices_list = []
    for ticker in tickers:
        try:
            print('Fetching data for %s' % str(ticker))
            prices = pdr.DataReader(ticker, 'yahoo', '01/01/2017')['Adj Close']
            prices = pd.DataFrame(prices)
            prices.columns = [ticker]
            prices_list.append(prices)
        except:
            pass
        prices_df = pd.concat(prices_list,axis=1)

    prices_df.sort_index(inplace=True)

    prices_df.head()

    # Calculate average annual percentage return and volatilities over a theoretical one year period
    returns = prices_df.pct_change().mean() * 252
    returns = pd.DataFrame(returns)
    returns.columns = ['Returns']
    returns['Volatility'] = prices_df.pct_change().std() * sqrt(252)

    # format the data as a numpy array to feed into the K-Means algorithm
    data = np.asarray([np.asarray(returns['Returns']), np.asarray(returns['Volatility'])]).T

    X = data
    distorsions = []
    for k in range(2, 20):
        k_means = KMeans(n_clusters=k)
        k_means.fit(X)
        distorsions.append(k_means.inertia_)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, 20), distorsions)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.show()
    plt.close()

    # computing K-Means with K = 5 (5 clusters)
    centroids, _ = kmeans(data, 5)
    # assign each sample to a cluster
    idx, _ = vq(data, centroids)

    # some plotting using numpy's logical indexing
    plot(data[idx == 0, 0], data[idx == 0, 1], 'ob',
         data[idx == 1, 0], data[idx == 1, 1], 'oy',
         data[idx == 2, 0], data[idx == 2, 1], 'or',
         data[idx == 3, 0], data[idx == 3, 1], 'og',
         data[idx == 4, 0], data[idx == 4, 1], 'om')
    plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=8)
    plt.show()
    plt.close()

    # identify the outlier
    print(returns.idxmax())

    # drop the relevant stock from our data
    returns.drop('BHF', inplace=True)

    # recreate data to feed into the algorithm
    data = np.asarray([np.asarray(returns['Returns']), np.asarray(returns['Volatility'])]).T

    # computing K-Means with K = 5 (5 clusters)
    centroids, _ = kmeans(data, 5)
    # assign each sample to a cluster
    idx, _ = vq(data, centroids)

    # some plotting using numpy's logical indexing
    plot(data[idx == 0, 0], data[idx == 0, 1], 'ob',
         data[idx == 1, 0], data[idx == 1, 1], 'oy',
         data[idx == 2, 0], data[idx == 2, 1], 'or',
         data[idx == 3, 0], data[idx == 3, 1], 'og',
         data[idx == 4, 0], data[idx == 4, 1], 'om')
    plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=8)
    plt.show()
    plt.close()

    details = [(name, cluster) for name, cluster in zip(returns.index, idx)]

    for detail in details:
        print(detail)

    plt.show()
    plt.close()
