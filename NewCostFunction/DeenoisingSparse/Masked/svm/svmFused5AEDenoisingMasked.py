import glob
from sklearn.model_selection import train_test_split  
from sklearn import svm
import pandas
from time import gmtime, strftime
import csv

users = [1,2,3,4,5,6]
activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]
features =  ["featuresFilt"]

for feature in features:

    for act in activities:
        sumFRR = 0;
        sumFAR = 0;
        
        
        with open("./5AEDenoisingMaskedResults/" + feature + "_MaskedRes_" + act + ".txt", "a") as myfile:
                myfile.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n\n\n")

        row_accuracy = []
        for us in users:
            activityType = act
            userNum = us
            featuresType = feature

            filenames = glob.glob('../resultsFusedDenoisingMasked5AE/AEResult_' + featuresType +'_' + activityType + '#*.csv')
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

            currentUserData = pandas.read_csv('../resultsFusedDenoisingMasked5AE/AEResult_' + featuresType +'_' + activityType + '#' + str(userNum) + '.csv', header = None)
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
            
            row_accuracy.append(str("%.2f" % ((totalCM[1][1]+totalCM[0][0])/(totalCM[1][1]+totalCM[0][0]+totalCM[0][1]+totalCM[1][0]))))

            
            with open("./5AEDenoisingMaskedResults/" + feature + "_MaskedRes_" + act + ".txt", "a") as myfile:
                myfile.write("User: " + str(us) + "\nFRR: " + str("%.5f" % FRR) + "\nFAR: " + str("%.5f" % FAR) + "\n\n\n")
        
        with open("results_new_accuracy/output.csv",'a') as f:
            writer = csv.writer(f, dialect='excel')
            writer.writerow(row_accuracy)   
        
        with open("./5AEDenoisingMaskedResults/" + feature + "_MaskedRes_" + act + ".txt", "a") as myfile:
                myfile.write("Mean: \nFRR: " + str("%.5f" % (sumFRR/6)) + "\nFAR: " + str("%.5f" % (sumFAR/6)) + "\n\n\n")


            
            