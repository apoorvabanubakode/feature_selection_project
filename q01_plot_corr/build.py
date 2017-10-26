# Default imports
import pandas as pd
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap
from matplotlib import pyplot as plt
data = pd.read_csv('data/house_prices_multivariate.csv')


# Write your solution here:


def plot_corr(data,size=11):
    corr=data.corr()

    fig, ax = plt.subplots(figsize=(size, size))
    plt.set_cmap('YlOrRd')
    ax.matshow(corr)
