# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


#data = pd.read_csv('data/house_prices_multivariate.csv')

def select_from_model(df):
    features=df.iloc[:,:-1]
    target=df.iloc[:,-1]

    select=SelectFromModel(RandomForestClassifier(random_state=9))
    x_new=select.fit_transform(features,target)
    support=select.get_support()
    #print support
    finalfeature=[]
    for i,feature in enumerate(features):
        if support[i]==True:
            finalfeature.append(feature)
    return finalfeature

# Your solution code here

# Your solution code here
