import glob
from sklearn.model_selection import train_test_split  
from sklearn import svm
import pandas
from time import gmtime, strftime
import csv
import os

activities = ["running", "walkingDownstairs", "walkingUpstairs"]
features =  ["featuresFilt"]

for feature in features:

    for act in activities:
        sumFRR = 0;
        sumFAR = 0;
        
        
        with open("./5AEDenoisingResults/" + feature + "_DenoisingRes_" + act + ".txt", "a") as myfile:
                myfile.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n\n\n")

        row_accuracy = []
        for us in range(56):
            activityType = act
            userNum = us+1
            featuresType = feature

            filepath = '../resultsFusedDenoising5AE/' + act + '/AEResult_' + featuresType +'_' + activityType + '#' + str(userNum) + '.csv'
            if os.path.isfile(filepath): 
            
                filenames = glob.glob('../resultsFusedDenoising5AE/' + act + '/AEResult_' + featuresType +'_' + activityType + '#*.csv')
    
                allUsersFeatures = pandas.DataFrame()
    
                for item in filenames:
                    # Load current dataset
                    url = item
                    if (url != filepath):
                        dataset = pandas.read_csv(url, header = None, engine='python')
                        allUsersFeatures = pandas.concat([allUsersFeatures, dataset], ignore_index=True)
    
                allUsersFeatures["target"] = -1
                impostors = allUsersFeatures["target"]
                allUsersFeatures.drop(["target"], axis=1, inplace=True)
    
                currentUserData = pandas.read_csv(filepath, header = None)
                currentUserData['target'] = 1
                
                curUserTarget = currentUserData['target']
                
                currentUserData.drop(["target"], axis=1, inplace=True)
                
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
                
                sumFRR = sumFRR + FRR
                sumFAR = sumFAR + FAR
                
                row_accuracy.append(str("%.2f" % (((totalCM[1][1]+totalCM[0][0])/(totalCM[1][1]+totalCM[0][0]+totalCM[0][1]+totalCM[1][0]))*100)))
    
                
                with open("./5AEDenoisingResults/" + feature + "_DenoisingRes_" + act + ".txt", "a") as myfile:
                    myfile.write("User: " + str(userNum) + "\nFRR: " + str("%.5f" % FRR) + "\nFAR: " + str("%.5f" % FAR) + "\n\n\n")
        
        with open('results_new_accuracy/output' + act + '.csv','a') as f:
            writer = csv.writer(f, dialect='excel')
            writer.writerow(row_accuracy)  
#            for item in row_accuracy:
#                writer.writerow([item,]) 
            
        amount_of_users = len(os.listdir('../resultsFusedDenoising5AE/' + act))
        
        with open("./5AEDenoisingResults/" + feature + "_DenoisingRes_" + act + ".txt", "a") as myfile:
                myfile.write("Mean: \nFRR: " + str("%.5f" % (sumFRR/amount_of_users)) + "\nFAR: " + str("%.5f" % (sumFAR/amount_of_users)) + "\n\n\n")
                
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

            
            