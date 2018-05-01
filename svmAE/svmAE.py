import glob
from sklearn.model_selection import train_test_split  
from sklearn import svm
import numpy as np
import pandas
from time import gmtime, strftime

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import csv

users = [1,2,3,4,5,6]
activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]
features =  ["featuresFilt"]

for feature in features:

    for act in activities:
        sumFRR = 0;
        sumFAR = 0;
        
        
        with open(feature + "_5DeepAEresult_" + act + ".txt", "a") as myfile:
                myfile.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n\n\n")
                
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        thresholds = dict()

        row_accuracy = []
        
        for us in users:
            activityType = act
            userNum = us
            featuresType = feature

            filenames = glob.glob('../AutoEncoderMyData/results5AEDeep/AEResult_' + featuresType +'_' + activityType + '#*.csv')
            del filenames[us-1]

            allUsersFeatures = pandas.DataFrame()

            for item in filenames:
                # Load current dataset
                url = item
                dataset = pandas.read_csv(url, header = None, engine='python')
                allUsersFeatures = pandas.concat([allUsersFeatures, dataset], ignore_index=True)

            allUsersFeatures["target"] = -1
            impostors = allUsersFeatures["target"]
            allUsersFeatures.drop(["target"], axis=1, inplace=True)

            currentUserData = pandas.read_csv('../AutoEncoderMyData/results5AEDeep/AEResult_' + featuresType +'_' + activityType + '#' + str(userNum) + '.csv', header = None)
            currentUserData['target'] = 1
            
            curUserTarget = currentUserData['target']
            
            currentUserData.drop(["target"], axis=1, inplace=True)
            
            train_data, test_data, train_target, test_target = train_test_split(currentUserData, curUserTarget, train_size = 0.8, test_size = 0.2)  
            
            model = svm.OneClassSVM(nu=0.1, kernel='linear')  
            
            test_data_with_impostors = np.r_[test_data, allUsersFeatures]
            y_score = model.fit(train_data).decision_function(test_data_with_impostors) 
            
            y_pred_train =  model.predict(train_data) 
            y_pred_test = model.predict(test_data)
            
            y_labels = np.array([1]*test_data.shape[0]+[-1]*allUsersFeatures.shape[0])
            
            fpr[us], tpr[us], thresholds[us] = roc_curve(y_labels, y_score)
            roc_auc[us] = auc(fpr[us], tpr[us])
            
                        
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
            
            sumFRR = sumFRR + FRR
            sumFAR = sumFAR + FAR
            
            row_accuracy.append(str("%.2f" % ((totalCM[1][1]+totalCM[0][0])/(totalCM[1][1]+totalCM[0][0]+totalCM[0][1]+totalCM[1][0]))))

            
            with open(feature + "_5DeepAEresult_" + act + ".txt", "a") as myfile:
                myfile.write("User: " + str(us) + "\nFRR: " + str("%.5f" % FRR) + "\nFAR: " + str("%.5f" % FAR) + "\n\n\n")
#        
        with open("results_new_accuracy/output.csv",'a') as f:
            writer = csv.writer(f, dialect='excel')
            writer.writerow(row_accuracy)   
        with open(feature + "_5DeepAEresult_" + act + ".txt", "a") as myfile:
                myfile.write("Mean: \nFRR: " + str("%.5f" % (sumFRR/6)) + "\nFAR: " + str("%.5f" % (sumFAR/6)) + "\n\n\n")
        
#        
#        feature_type = "Original" if feature == "featuresOrig" else "Filtered"
#        
#        plt.figure()
#        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'lime', 'indigo', 'crimson'])
#        for i, color in zip(users, colors):
#            plt.plot(fpr[i], tpr[i], color=color, label='user {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
#
#        plt.plot([0, 1], [0, 1], 'k--')
#        plt.xlim([0.0, 1.0])
#        plt.ylim([0.0, 1.05])
#        plt.xlabel('False Positive Rate')
#        plt.ylabel('True Positive Rate')
#        plt.rc('axes', titlesize=15)     # fontsize of the axes title
#        plt.rc('axes', labelsize=15)
#        plt.rc('legend', fontsize=15) 
#        plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
#        plt.rc('ytick', labelsize=13)
##        plt.title('ROCs for {0} obtained from {1} features'.format(act, feature_type))
#        plt.legend(loc="lower right")
#        plt.tight_layout()
#        plt.savefig('./rocs/' + feature + '_' + act + '_result.jpg', format='jpg')
#        plt.close()
                
#            curColumn[act][counter] = [FRR, FAR]
#            counter=counter+1
#        
#        sumFRR = 0;
#        sumFAR = 0;
#        
#        for i in [0,1,2,3,4,5]:
#            sumFRR = sumFRR + curColumn[act][i][0]
#            sumFAR = sumFAR + curColumn[act][i][1]
#            
#        curColumn[act][0][0]
#        
#        curColumn[act][6] = [sumFRR/6, sumFAR/6]
#
#    last = errTable.index[-1]
#    errTable = errTable.rename(index={last: 'mean'})
#    
#    errTable.to_csv('E:/Study/ThesisGit/Thesis/svmAuth/svmAuth' + feature+ '_results.csv', index = True)

            
            