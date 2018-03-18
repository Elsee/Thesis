from sklearn.model_selection import train_test_split  
import pandas
import os

def write_imposters(impostData, experiment, activity, user, featuresType):
    imposterData = impostData.sample(frac=1).reset_index(drop=True)
    
    train_imp_data, test_imp_data = train_test_split(imposterData, train_size = 0.2, test_size = 0.8)  
    for index, row in train_imp_data.iterrows() :
        my_str = "-1 "
        for i in range(imposterData.shape[1]):
            my_str = my_str + str(i+1) + ":" + str(row[i]) + " "
        my_str = my_str + "\n"
        
        with open('./test'+str(experiment)+ '/'+activity+'/'+str(user)+'/training_data_' + featuresType + '.data', "a") as myfile:
            myfile.write(my_str)
            
    for index, row in test_imp_data.iterrows() :
        my_str = "-1 "
        for i in range(imposterData.shape[1]):
            my_str = my_str + str(i+1) + ":" + str(row[i]) + " "
        my_str = my_str + "\n"
        
        with open('./test'+str(experiment)+ '/'+activity+'/'+str(user)+'/testing_data_' + featuresType + '.data', "a") as myfile:
            myfile.write(my_str)
            
def write_rest_imposters_data(imp_list, rest_imp_list, experiment, activity, user, featuresType):
    rest_imp = [x for x in imp_list if x not in rest_imp_list]
                    
    restImpData = pandas.DataFrame()
    
    for rest_imp_member in rest_imp:
        restImpMemberData = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(rest_imp_member) + '.csv', header = 0)
        restImpData = pandas.concat([restImpData, restImpMemberData], ignore_index=True)  

    for index, row in restImpData.iterrows() :
        my_str = "-1 "
        for i in range(restImpData.shape[1]):
            my_str = my_str+str(i+1) + ":" + str(row[i]) + " "
        my_str = my_str+"\n"   

        with open('./test'+str(experiment)+ '/'+activity+'/'+str(user)+'/testing_data_' + featuresType + '.data', "a") as myfile:
            myfile.write(my_str) 



users = [1,2,3,4,5,6]
activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]
features =  ["featuresFilt"]
EXPERIMENTS = 1

for feature in features:
    for act in activities:        
        for us in users:
            directory = "./dataToTest"
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = "./dataToTest/" + act
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = "./dataToTest/" + act + "/" + str(us)
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            directory = "./dataToTest/" + act + "/" + str(us) + "/" + "results"
            if not os.path.exists(directory):
                os.makedirs(directory)
        
            activityType = act
            userNum = us
            featuresType = feature
            
            currentUserData = pandas.read_csv('../resultsFusedDenoising5AE/AEResult_' + featuresType +'_' + activityType + '#' + str(userNum) + '.csv', header = 0)
            
#           Prepare a list of all other users
            other_users = list(users)
            other_users.remove(us)
            
            for exp in range(EXPERIMENTS):
                exp_num = exp+1
                open('./dataToTest/'+act+'/'+str(us)+'/training_data_' + featuresType + '.data', 'w').close()
                open('./dataToTest/'+act+'/'+str(us)+'/testing_data_' + featuresType + '.data', 'w').close()
                
                train_data, test_data = train_test_split(currentUserData, train_size = 0.5, test_size = 0.5)  

                for index, row in train_data.iterrows() :
                    my_str = "1 "
                    for i in range(train_data.shape[1]):
                        my_str = my_str+str(i+1) + ":" + str(row[i]) + " "
                    my_str = my_str+"\n"
                    with open('./dataToTest/'+act+'/'+str(us)+'/training_data_' + featuresType + '.data', "a") as myfile:
                            myfile.write(my_str)
                            
                for index, row in test_data.iterrows() :
                    my_str = "1 "
                    for i in range(test_data.shape[1]):
                        my_str = my_str+str(i+1) + ":" + str(row[i]) + " "
                    my_str = my_str+"\n"
                    with open('./dataToTest/'+act+'/'+str(us)+'/testing_data_' + featuresType + '.data', "a") as myfile:
                            myfile.write(my_str)
                
                restImpData = pandas.DataFrame()
                
                for other_users_member in other_users:
                    restImpMemberData = pandas.read_csv('../resultsFusedDenoising5AE/AEResult_' + featuresType +'_' + activityType + '#' + str(other_users_member) + '.csv', header = 0)
                    restImpData = pandas.concat([restImpData, restImpMemberData], ignore_index=True)  
            
                for index, row in restImpData.iterrows() :
                    my_str = "-1 "
                    for i in range(restImpData.shape[1]):
                        my_str = my_str+str(i+1) + ":" + str(row[i]) + " "
                    my_str = my_str+"\n"   
            
                    with open('./dataToTest/'+act+'/'+str(us)+'/testing_data_' + featuresType + '.data', "a") as myfile:
                        myfile.write(my_str) 