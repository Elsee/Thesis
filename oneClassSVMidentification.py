import glob
from sklearn.model_selection import train_test_split  
from sklearn import svm
import numpy as np
import pandas
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

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
            
            allUsersFeatures.loc[allUsersFeatures['user'] == userNum, "attack"] = 1  
            allUsersFeatures.loc[allUsersFeatures['user'] != userNum, "attack"] = -1
            
            target = allUsersFeatures['attack']
            target1 = label_binarize(target, classes=[0, 1])
            n_classes = target1.shape[1]
            
            targetUser = target[target == 1]
            outliers = target[target == -1]  
            
            allUsersFeatures.drop(["user", "attack"], axis=1, inplace=True)
             # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            allUsersFeatures = sc.fit_transform(allUsersFeatures)
            
            train_data, test_data, train_target, test_target = train_test_split(allUsersFeatures, target, train_size = 0.8, test_size = 0.2)  

            model = svm.OneClassSVM(nu=0.5, kernel='rbf', gamma=0.00005)  
            y_score = model.fit(train_data).decision_function(test_data)
            
            preds = model.predict(train_data)  
            
            
            preds1 = model.predict(allUsersFeatures)
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(target, preds1)
            
            targUs = preds1[:len(targetUser)] 
            impostors = preds1[:len(outliers)]  
            
            FRR = cm[1][0] / (cm[1][0] + cm[1][1])
            FAR = cm[0][1] / (cm[0][0] + cm[0][1])
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(test_target[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
             # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(test_target.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            plt.figure()
            lw = 2
            plt.plot(fpr[0], tpr[0], color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curve for user : ' + str(userNum) + ' doing activity: ' + activityType + ' recieved from ' + featuresType)
            plt.legend(loc="lower right")
            plt.savefig('E:/Study/ThesisGit/Thesis/oneClassSVMident/ROC_curves/' + featuresType +'_' + activityType + str(userNum) + '.png', bbox_inches='tight')

            plt.show()
            plt.close()
            
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
    
    errTable.to_csv('E:/Study/ThesisGit/Thesis/oneClassSVMident/oneClassSVMident' + feature+ '_results.csv', index = True)