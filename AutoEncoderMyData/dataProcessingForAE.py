import pandas as pd
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

filenames = glob.glob('../myTrainingData/featuresOrig_*.csv')

totalData = pd.DataFrame()
    
for item in filenames:
    # Load current dataset
    url = item
    #choose only accelerometer data
    data = pd.read_csv(url, header = 0, engine='python')
    totalData = pd.concat([totalData, data], ignore_index=True)

totalData = totalData.sort_values(['user'], ascending = 1)

#totalData.set_index(keys=['user'], drop=False,inplace=True)
#labels=totalData['user'].unique().tolist()
#
#usersData = {}
#usersDataLen = []
#
#for i in labels:
#     usersData["user{0}".format(i)] = totalData.loc[totalData.user==i]
#
#     usersDataLen.append(usersData["user{0}".format(i)].shape[0])
#     
#minUserSamples = min(usersDataLen)
#segments = []
#sc = StandardScaler()
#
#for i in labels:
#    usersData["user{0}".format(i)] = usersData["user{0}".format(i)].head(n=minUserSamples)
#    usersData["user{0}".format(i)] = sc.fit_transform(usersData["user{0}".format(i)])
#
#    segments.append(np.array(usersData["user{0}".format(i)][:,:-1]))
#
#labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)
sc = StandardScaler()

numRows = 57
step = 20
segments = []
labels = []

for i in range(0, len(totalData) - numRows, step):
    label = stats.mode(totalData['user'][i: i + numRows])[0][0]
    segments.append(np.array(sc.fit_transform(totalData.iloc[i: i + numRows,:-1])))
    labels.append(label)
    
segments = np.asarray(segments, dtype= np.float32)
labels = np.asarray(labels, dtype= np.float32)