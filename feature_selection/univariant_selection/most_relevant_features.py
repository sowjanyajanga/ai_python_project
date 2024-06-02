MSN_MAPPINGS= {'CLEIEUS' : 10,
               'DKEIEUS' : 9,
               'PAEIEUS' : 8,
               'PCEIEUS' : 7,
               'RFEIEUS' : 6,
               'TXEIEUS' : 5,
               'NWEIEUS':  2,
               'NNEIEUS' : 1,
               'GEEIEUS' : 0
}


def convertMSN(msn):
    if msn is not None and MSN_MAPPINGS[msn] is not None:
        return MSN_MAPPINGS[msn]
    else:
        print('--' + msn + '--')
        raise Exception('Found an unknown msn type')


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = pd.read_csv('../data/MER_T12_06.csv')

data['MSN']=  data['MSN'].apply(convertMSN)
data['Description']=  data['Description'].apply(lambda x : abs(hash(x)))
data['Unit']=  data['Unit'].apply(hash)

X= data[['MSN','YYYYMM','Column_Order','Description','Unit']].iloc[:,:]
print(X)

y= data[['Value']].iloc[:,:]

# X = data.iloc[:,0:20] #independent columns
# y = data.iloc[:,-1] #pick last column for the target feature
#apply SelectKBest class to extract top 5 best features
bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
scores = pd.concat([dfcolumns,dfscores],axis=1)
scores.columns = ['specs','score']
print(scores.nlargest(5,'score')) #print the 5 best features