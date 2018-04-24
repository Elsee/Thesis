from sklearn.model_selection import train_test_split  
from sklearn import svm
import pandas
from time import gmtime, strftime
import csv
import os
from sklearn.model_selection import KFold
import numpy as np

activities = ["running", "walkingDownstairs", "walkingUpstairs", "walking"]
features =  ["featuresFilt"]

for feature in features:

    for act in activities:
        sumFRR = 0;
        sumFAR = 0;
        
        
        with open("./5AEDenoisingResults/" + feature + "_DenoisingRes_" + act + ".txt", "a") as myfile:
                myfile.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n\n\n")

        row_accuracy = [[],[],[],[],[]]
        for us in range(56):
            activityType = act
            userNum = us+1
            featuresType = feature

            filepath = 'AEResult_' + featuresType +'_' + activityType + '#' + str(userNum) + '.csv'
            if os.path.isfile('../resultsFusedDenoising5AE/' + act + '/' + filepath): 
            
#                filenames = glob.glob('../resultsFusedDenoising5AE/' + act + '/AEResult_' + featuresType +'_' + activityType + '#*.csv')
                filenames = os.listdir('../resultsFusedDenoising5AE/' + act) 
                allUsersFeatures = pandas.DataFrame(columns=range(57))
    
                for item in filenames:
                    # Load current dataset
                    url = item
                    if (url != filepath):
                        dataset = pandas.read_csv('../resultsFusedDenoising5AE/' + act + '/' + url, header = None, engine='python')
#                        allUsersFeatures = pandas.concat([allUsersFeatures, dataset], ignore_index=True)
                        allUsersFeatures = np.r_[allUsersFeatures, dataset]
    
                allUsersFeatures = pandas.DataFrame(allUsersFeatures)
                allUsersFeatures["target"] = -1
                impostors = allUsersFeatures["target"]
                allUsersFeatures.drop(["target"], axis=1, inplace=True)
    
                currentUserData = pandas.read_csv('../resultsFusedDenoising5AE/' + act + '/' + filepath, header = None)
                currentUserData['target'] = 1
                
                curUserTarget = currentUserData['target']
                
                currentUserData.drop(["target"], axis=1, inplace=True)

                if (currentUserData.shape[0] > 4):

                    kf = KFold(n_splits=5)

                    with open("./5AEDenoisingResults/" + feature + "_DenoisingRes_" + act + ".txt", "a") as myfile:
                            myfile.write("RESULTS FOR USER: " + str(userNum) + "\n")

                    k_order = 0

                    for train, test in kf.split(currentUserData):
                        train_data, test_data = currentUserData.iloc[train], currentUserData.iloc[test]
                        train_target, test_target = curUserTarget.iloc[train], curUserTarget.iloc[test]
                    
                        model = svm.OneClassSVM(nu=0.1, kernel='linear')  
                        y_score = model.fit(train_data).decision_function(test_data) 
                        
                        y_pred_train =  model.predict(train_data) 
                        y_pred_test = model.predict(test_data)
                        
                                    
                         # Making the Confusion Matrix
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(test_target, y_pred_test)
                        
                        y_pred_outliers = model.predict(allUsersFeatures)
                        
                        cm1 = confusion_matrix(impostors, y_pred_outliers)
                        if (cm1.shape[1] != 2):
                            cm1 = np.append(cm1, [[0]], axis=1)
                        if (cm1.shape[0] != 2):
                            cm1 = np.append(cm1, [[0,0]], axis=0)
                        if (cm.shape[1] != 2):
                            cm = np.append([[0]], cm, axis=1)
                        if (cm.shape[0] != 2):
                            cm = np.append([[0,0]], cm, axis=0)
                        totalCM = cm + cm1
                        
                        n_error_train = y_pred_train[y_pred_train == -1].size
                        n_error_test = y_pred_test[y_pred_test == -1].size
                        n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
                        
                        FRR = totalCM[1][0] / (totalCM[1][0] + totalCM[1][1])
                        FAR = totalCM[0][1] / (totalCM[0][0] + totalCM[0][1])
                        
                        sumFRR = sumFRR + FRR
                        sumFAR = sumFAR + FAR
                        
                        row_accuracy[k_order].append(str("%.2f" % (((totalCM[1][1]+totalCM[0][0])/(totalCM[1][1]+totalCM[0][0]+totalCM[0][1]+totalCM[1][0]))*100)))

                        k_order = k_order + 1
                        
                        with open("./5AEDenoisingResults/" + feature + "_DenoisingRes_" + act + ".txt", "a") as myfile:
                            myfile.write("k-fold: " + str(k_order) + "\nFRR: " + str("%.5f" % FRR) + "\nFAR: " + str("%.5f" % FAR) + "\n")
        
        with open('results_new_accuracy/output' + act + '.csv','a') as f:   
            writer = csv.writer(f, dialect='excel')
            for line in row_accuracy:
                writer.writerow(line)
#            for item in row_accuracy:
#                writer.writerow([item,]) 
            
        # amount_of_users = len(os.listdir('../resultsFusedDenoising5AE/' + act))
        
        # with open("./5AEDenoisingResults/" + feature + "_DenoisingRes_" + act + ".txt", "a") as myfile:
        #         myfile.write("Mean: \nFRR: " + str("%.5f" % (sumFRR/amount_of_users)) + "\nFAR: " + str("%.5f" % (sumFAR/amount_of_users)) + "\n\n\n")
                
#            curColumn[act][counter] = [FRR, FAR]
#            counter=counter+1
#        
#        sumFRR = 0;
#        sumFAR = 0;
#        
#        for i in [0,1,2,3,4,5]:
#            sumFAR = sumFAR + curColumn[act][i][1]
#            
#            sumFRR = sumFRR + curColumn[act][i][0]
#        curColumn[act][0][0]
#        
#        curColumn[act][6] = [sumFRR/6, sumFAR/6]
#
#    last = errTable.index[-1]
#    errTable = errTable.rename(index={last: 'mean'})
#    
#    errTable.to_csv('E:/Study/ThesisGit/Thesis/svmAuth/svmAuth' + feature+ '_results.csv', index = True)

            
            