print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import pandas

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split  

logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

activityType = "Walking"
userNum = 1
featuresType = "featuresOrig"
            
currentUserData = pandas.read_csv('D:/Study/Thesis/System/myTrainingData/' + featuresType +'_' + activityType + str(userNum) + '.csv', header = None, skiprows = 1)
currentUserData.drop([57], axis=1, inplace=True)

train_data = currentUserData
test_data = currentUserData.iloc[:, 56]

# Plot the PCA spectrum
pca.fit(train_data)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

# Prediction
n_components = [20, 40, 57]
Cs = np.logspace(-4, 4, 3)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(train_data, (test_data*10).astype(int))

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()