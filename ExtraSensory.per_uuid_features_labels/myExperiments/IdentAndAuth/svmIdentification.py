# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

# Importing the dataset

activities = ["walking"]
features =  ["featuresFilt"]

for feature in features:
    
    for act in activities:
        sumFRR = 0;
        sumFAR = 0;
        
        activityType = act
        featuresType = feature

        filenames = glob.glob('../extractedFeatures/' + featuresType +'_' + activityType + '#*.csv')
        dataset = pd.DataFrame()
        
        row_accuracy = []
        
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
        classifier = SVC(C=10, kernel = 'linear', random_state = 0)
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
            
            with open('./svmIdent/' + feature + "svmIdent" + act + ".txt", "a") as myfile:
                myfile.write("User: " + str(i) + "\nFRR: " + str("%.5f" % FRR) + "\nFAR: " + str("%.5f" % FAR) + "\n\n\n") 
            
            sumFRR = sumFRR + FRR
            sumFAR = sumFAR + FAR          

            row_accuracy.append(str("%.2f" % (((totalCM[1][1]+totalCM[0][0])/(totalCM[1][1]+totalCM[0][0]+totalCM[0][1]+totalCM[1][0]))*100)))
                                              
    
    
        with open('./svmIdent/' + feature + "svmIdent" + act + ".txt", "a") as myfile:
                myfile.write("Mean: \nFRR: " + str("%.5f" % (sumFRR/6)) + "\nFAR: " + str("%.5f" % (sumFAR/6)) + "\n\n\n")
    
