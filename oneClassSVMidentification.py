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

            filenames = glob.glob('D:/Study/Thesis/System/myTrainingData/' + featuresType +'_' + activityType + '*.csv')
            allUsersFeatures = pandas.DataFrame()
                
            for item in filenames:
                # Load current dataset
                url = item
                dataset = pandas.read_csv(url, header = 0, engine='python')
                allUsersFeatures = pandas.concat([allUsersFeatures, dataset], ignore_index=True)
            
            allUsersFeatures.loc[allUsersFeatures['user'] == userNum, "attack"] = 1  
            allUsersFeatures.loc[allUsersFeatures['user'] != userNum, "attack"] = -1
            
            target = allUsersFeatures['attack']
            
            targetUser = target[target == 1]
            outliers = target[target == -1]  
            
            allUsersFeatures.drop(["user", "attack"], axis=1, inplace=True)
             # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            allUsersFeatures = sc.fit_transform(allUsersFeatures)
            
            train_data, test_data, train_target, test_target = train_test_split(allUsersFeatures, target, train_size = 0.8, test_size = 0.2)  

            model = svm.OneClassSVM(nu=0.5, kernel='rbf', gamma=0.00005)  
            model.fit(train_data)
            
            preds = model.predict(train_data)  
            
            
            preds1 = model.predict(allUsersFeatures)
            targUs = preds1[:len(targetUser)] 
            impostors = preds1[:len(outliers)]  
            FRR = targUs[targUs == -1].size / targUs.size * 100
            FAR = impostors[impostors == 1].size / impostors.size * 100
            
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
    
    errTable.to_csv('D:/Study/Thesis/System/svmIdent/' + feature+ '_results.csv', index = True)