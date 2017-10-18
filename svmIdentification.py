# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

# Importing the dataset

activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]
features =  ["featuresOrig", "featuresFilt", "featuresOrigPCA40", "featuresOrigPCA57", "featuresFiltPCA40", "featuresFiltPCA57"]

for feature in features:
    errTable = pd.DataFrame()
    
    for act in activities:
        
        activityType = act
        featuresType = feature

        filenames = glob.glob('E:/Study/ThesisGit/Thesis/myTrainingData/' + featuresType +'_' + activityType + '*.csv')
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
        
        user_number = cm.shape[0]
        
        curColumn = pd.DataFrame(np.zeros((7,1)), columns={act})
        curColumn = curColumn.astype('object')
        
        for i in range(0, user_number):
            FRR = (cm[i,:].sum() - cm[i][i])/cm[i,:].sum()
            FAR = (cm[:,i].sum() - cm[i][i])/(cm[:,:].sum() - cm[i,:].sum())
            
            curColumn[act][i] = [FRR, FAR]
            
        sumFRR = 0;
        sumFAR = 0;
        
        for i in range(0, user_number):
            sumFRR = sumFRR + curColumn[act][i][0]
            sumFAR = sumFAR + curColumn[act][i][1]
            
        curColumn[act][0][0]
        
        curColumn[act][6] = [sumFRR/6, sumFAR/6]
        
        errTable = errTable.join(curColumn, how='outer')

    last = errTable.index[-1]
    errTable = errTable.rename(index={last: 'mean'})
    
    errTable.to_csv('E:/Study/ThesisGit/Thesis/svmIdent/svmIdent' + feature+ '_results.csv', index = True)
    
