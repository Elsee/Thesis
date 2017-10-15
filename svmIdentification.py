# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

# Importing the dataset

filenames = glob.glob('D:/Study/Thesis/System/myTrainingData/featuresOrig_Walking*.csv')
dataset = pd.DataFrame()
    
for item in filenames:
    # Load current dataset
    url = item
    data = pd.read_csv(url, header = 0, engine='python')
    dataset = pd.concat([dataset, data], ignore_index=True)

y = dataset['user']
dataset.drop(["user"], axis=1, inplace=True)
X = dataset

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)