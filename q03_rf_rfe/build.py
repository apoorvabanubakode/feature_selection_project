# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df):
    df_features=df.iloc[:,:-1]
    df_target=df.iloc[:,-1]
    df_header=list(df)
    n= (len(df_header)-1)/2
    final_features=[]
    model=RFE(RandomForestClassifier(),n_features_to_select=n)
    x_new=model.fit_transform(df_features,df_target)
    #print(model.n_features_)
    feature_filter=(model.support_)
    a=0
    for i,header in enumerate(df_header[:-1]):
        if feature_filter[i]==True:
            final_features.append(header)


    return(final_features)

# Your solution code here
