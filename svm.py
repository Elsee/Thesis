import glob
from sklearn.model_selection import train_test_split  
from sklearn import svm
import numpy as np
import pandas

users = [1,2,3,4,5,6]
activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]
features =  ["featuresOrig", "featuresFilt", "featuresOrigPCA40", "featuresOrigPCA57", "featuresFiltPCA40", "featuresFiltPCA57"]

for feature in features:
    errTable = pandas.DataFrame()
    
    for act in activities:
        curColumn = pandas.DataFrame(np.zeros((7,1)), columns={act})
        curColumn = curColumn.astype('object')
        counter = 0
        
        for us in users:
            activityType = act
            userNum = us
            featuresType = feature
    
            filenames = glob.glob('E:/Study/ThesisGit/Thesis/myTrainingData/' + featuresType +'_' + activityType + '*.csv')
    
            allUsersFeatures = pandas.DataFrame()
    
            for item in filenames:
                # Load current dataset
                url = item
                dataset = pandas.read_csv(url, header = 0, engine='python')
                allUsersFeatures = pandas.concat([allUsersFeatures, dataset], ignore_index=True)
        
            allUsersFeatures.drop(allUsersFeatures[allUsersFeatures.user == userNum].index, inplace=True)
            allUsersFeatures["user"] = -1
            impostors = allUsersFeatures["user"]
            allUsersFeatures.drop(["user"], axis=1, inplace=True)
          
            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            allUsersFeatures = sc.fit_transform(allUsersFeatures)
            
            currentUserData = pandas.read_csv('E:/Study/ThesisGit/Thesis/myTrainingData/' + featuresType +'_' + activityType + str(userNum) + '.csv', header = 0)
            currentUserData['target'] = 1
            
            curUserTarget = currentUserData['target']
            
            currentUserData.drop(["user", "target"], axis=1, inplace=True)
            
            currentUserData = sc.fit_transform(currentUserData)
            
            train_data, test_data, train_target, test_target = train_test_split(currentUserData, curUserTarget, train_size = 0.8, test_size = 0.2)  
            
            model = svm.OneClassSVM(nu=0.1, kernel='linear')  
            y_score = model.fit(train_data).decision_function(test_data) 
            
            y_pred_train =  model.predict(train_data) 
            y_pred_test = model.predict(test_data)
            
                        
             # Making the Confusion Matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(test_target, y_pred_test)
            
            y_pred_outliers = model.predict(allUsersFeatures)
            
            cm1 = confusion_matrix(impostors, y_pred_outliers)
            totalCM = cm + cm1
            
            n_error_train = y_pred_train[y_pred_train == -1].size
            n_error_test = y_pred_test[y_pred_test == -1].size
            n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
            
            FRR = totalCM[1][0] / (totalCM[1][0] + totalCM[1][1])
            FAR = totalCM[0][1] / (totalCM[0][0] + totalCM[0][1])
            
            curColumn[act][counter] = [FRR, FAR]
            counter=counter+1
        
        sumFRR = 0;
        sumFAR = 0;
        
        for i in [0,1,2,3,4,5]:
            sumFRR = sumFRR + curColumn[act][i][0]
            sumFAR = sumFAR + curColumn[act][i][1]
            
        curColumn[act][0][0]
        
        curColumn[act][6] = [sumFRR/6, sumFAR/6]
        
        errTable = errTable.join(curColumn, how='outer')

    last = errTable.index[-1]
    errTable = errTable.rename(index={last: 'mean'})
    
    errTable.to_csv('E:/Study/ThesisGit/Thesis/svmAuth/svmAuth' + feature+ '_results.csv', index = True)

            
            