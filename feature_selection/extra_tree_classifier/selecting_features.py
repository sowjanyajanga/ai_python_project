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
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('../data/MER_T12_06.csv')

# X = data.iloc[:,0:20] #independent columns
# y = data.iloc[:,-1] # pick last column for the target feature

X= data[['MSN','YYYYMM','Column_Order']]
# Convert so that
# test = X.groupby('MSN',  group_keys=False).groups.keys()
# print(test)
# test = X.groupby('Unit',  group_keys=False).groups.keys()
# print(test)
# exit()

X['MSN']=  X['MSN'].apply(convertMSN)
X1=X.iloc[:,:] # To convert the column names to number indexes
print(X)
y= data[['Value']]
y1= y.iloc[:,:] # To convert the column names to number indexes
model = ExtraTreesClassifier()
model.fit(X1,y1)
print(model.feature_importances_) #use inbuilt class
#feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

