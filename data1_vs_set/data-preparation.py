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
features =  ["featuresOrig", "featuresFilt"]
EXPERIMENTS = 31

for feature in features:
    for act in activities:        
        for us in users:
            
            for dir_num in range(EXPERIMENTS):
                act_dir_num = dir_num + 1
                directory = "./test" + str(act_dir_num)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                directory = "./test" + str(act_dir_num) + "/" + act
                if not os.path.exists(directory):
                    os.makedirs(directory)
                directory = "./test" + str(act_dir_num) + "/" + act + "/" + str(us)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    
            directory = "./test1class"
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = "./test1class/" + act
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = "./test1class/" + act + "/" + str(us)
            if not os.path.exists(directory):
                os.makedirs(directory)
        
            activityType = act
            userNum = us
            featuresType = feature
            
            currentUserData = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(userNum) + '.csv', header = 0)
            currentUserData.drop(["user"], axis=1, inplace=True)
            
#           Prepare a list of all other users
            other_users = list(users)
            other_users.remove(us)
            
            for exp in range(EXPERIMENTS):
                exp_num = exp+1
                open('./test'+str(exp_num)+ '/'+act+'/'+str(us)+'/training_data_' + featuresType + '.data', 'w').close()
                open('./test'+str(exp_num)+ '/'+act+'/'+str(us)+'/testing_data_' + featuresType + '.data', 'w').close()
                
#               Prepare positive data
                for index, row in currentUserData.iterrows() :
                    half_of_data = currentUserData.shape[0]/2 if currentUserData.shape[0]%2 == 0 else currentUserData.shape[0]/2+1
                    my_str = "1 "
                    for i in range(currentUserData.shape[1]):
                        my_str = my_str+str(i+1) + ":" + str(row[i]) + " "
                    my_str = my_str+"\n"
                        
                    if (index<half_of_data):
                        with open('./test'+str(exp_num)+ '/'+act+'/'+str(us)+'/training_data_' + featuresType + '.data', "a") as myfile:
                            myfile.write(my_str)
            
                        if (exp_num==1):
                            with open('./test1class/'+act+'/'+str(us)+'/training_data_1class' + featuresType + '.data', "a") as myfile:
                                myfile.write(my_str)

                    else:
                        with open('./test'+str(exp_num)+ '/'+act+'/'+str(us)+'/testing_data_' + featuresType + '.data', "a") as myfile:
                            myfile.write(my_str)
                
                if (exp_num==1 or exp_num==2 or exp_num==3 or exp_num==4 or exp_num==5): 
                    imp = other_users[exp_num-1]
                    impData = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(imp) + '.csv', header = 0)
                    impData.drop(["user"], axis=1, inplace=True)
                    
                    write_imposters(impData, exp_num, act, us, featuresType)
                    
                    write_rest_imposters_data(other_users, [imp], exp_num, act, us, featuresType)                          
                
                elif (exp_num==6 or exp_num==7 or exp_num==8 or exp_num==9 or exp_num==10 or exp_num==11 or exp_num==12 or exp_num==13 or exp_num==14 or exp_num==15): 
                    if (exp_num==6):
                        imp = [other_users[0],other_users[1]]
                    elif (exp_num==7):
                        imp = [other_users[0],other_users[2]]
                    elif (exp_num==8):
                        imp = [other_users[0],other_users[3]]
                    elif (exp_num==9):
                        imp = [other_users[0],other_users[4]]
                    elif (exp_num==10):
                        imp = [other_users[1],other_users[2]]
                    elif (exp_num==11):
                        imp = [other_users[1],other_users[3]]
                    elif (exp_num==12):
                        imp = [other_users[1],other_users[4]]
                    elif (exp_num==13):
                        imp = [other_users[2],other_users[3]]
                    elif (exp_num==14):
                        imp = [other_users[2],other_users[4]]
                    elif (exp_num==15):
                        imp = [other_users[3],other_users[4]]
                        
                    impData1 = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(imp[0]) + '.csv', header = 0)
                    impData1.drop(["user"], axis=1, inplace=True)
                    impData2 = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(imp[1]) + '.csv', header = 0)
                    impData2.drop(["user"], axis=1, inplace=True)
                    
                    impData = pandas.concat([impData1, impData2], ignore_index=True)
                        
                    write_imposters(impData, exp_num, act, us, featuresType)
                    
                    write_rest_imposters_data(other_users, imp, exp_num, act, us, featuresType)
                        
                elif (exp_num==16 or exp_num==17 or exp_num==18 or exp_num==19 or exp_num==20 or exp_num==21 or exp_num==22 or exp_num==23 or exp_num==24 or exp_num==25): 
                    if (exp_num==16):
                        imp = [other_users[0],other_users[1],other_users[2]]
                    elif (exp_num==17):
                        imp = [other_users[0],other_users[1],other_users[3]]
                    elif (exp_num==18):
                        imp = [other_users[0],other_users[1],other_users[4]]
                    elif (exp_num==19):
                        imp = [other_users[0],other_users[2],other_users[3]]
                    elif (exp_num==20):
                        imp = [other_users[0],other_users[2],other_users[4]]
                    elif (exp_num==21):
                        imp = [other_users[0],other_users[3],other_users[4]]
                    elif (exp_num==22):
                         imp = [other_users[1],other_users[2],other_users[3]]
                    elif (exp_num==23):
                        imp = [other_users[1],other_users[2],other_users[4]]
                    elif (exp_num==24):
                        imp = [other_users[1],other_users[3],other_users[4]]
                    elif (exp_num==25):
                        imp = [other_users[2],other_users[3],other_users[4]]
                    
                    impData1 = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(imp[0]) + '.csv', header = 0)
                    impData1.drop(["user"], axis=1, inplace=True)
                    impData2 = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(imp[1]) + '.csv', header = 0)
                    impData2.drop(["user"], axis=1, inplace=True)
                    impData3 = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(imp[2]) + '.csv', header = 0)
                    impData3.drop(["user"], axis=1, inplace=True)
                    
                    impData = pandas.concat([impData1, impData2, impData3], ignore_index=True)
                        
                    write_imposters(impData, exp_num, act, us, featuresType)
                    
                    write_rest_imposters_data(other_users, imp, exp_num, act, us, featuresType)
                    
                elif (exp_num==26 or exp_num==27 or exp_num==28 or exp_num==29 or exp_num==30): 
                    if (exp_num==26):
                        imp = [other_users[0],other_users[1],other_users[2],other_users[3]]
                    elif (exp_num==27):
                        imp = [other_users[0],other_users[1],other_users[2],other_users[4]]
                    elif (exp_num==28):
                        imp = [other_users[0],other_users[1],other_users[3],other_users[4]]
                    elif (exp_num==29):
                        imp = [other_users[0],other_users[2],other_users[3],other_users[4]]
                    elif (exp_num==30):
                        imp = [other_users[1],other_users[2],other_users[3],other_users[4]]
                    
                    impData1 = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(imp[0]) + '.csv', header = 0)
                    impData1.drop(["user"], axis=1, inplace=True)
                    impData2 = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(imp[1]) + '.csv', header = 0)
                    impData2.drop(["user"], axis=1, inplace=True)
                    impData3 = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(imp[2]) + '.csv', header = 0)
                    impData3.drop(["user"], axis=1, inplace=True)
                    impData4 = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(imp[3]) + '.csv', header = 0)
                    impData4.drop(["user"], axis=1, inplace=True)
                    
                    impData = pandas.concat([impData1, impData2, impData3, impData4], ignore_index=True)
                        
                    write_imposters(impData, exp_num, act, us, featuresType)
                    
                    
                elif (exp_num==31):
                    imp = list(other_users)
                    
                    impData1 = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(imp[0]) + '.csv', header = 0)
                    impData1.drop(["user"], axis=1, inplace=True)
                    impData2 = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(imp[1]) + '.csv', header = 0)
                    impData2.drop(["user"], axis=1, inplace=True)
                    impData3 = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(imp[2]) + '.csv', header = 0)
                    impData3.drop(["user"], axis=1, inplace=True)
                    impData4 = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(imp[3]) + '.csv', header = 0)
                    impData4.drop(["user"], axis=1, inplace=True)
                    impData5 = pandas.read_csv('../myTrainingData/' + featuresType +'_' + activityType + '#' + str(imp[4]) + '.csv', header = 0)
                    impData5.drop(["user"], axis=1, inplace=True)
                    
                    impData = pandas.concat([impData1, impData2, impData3, impData4], ignore_index=True)
                        
                    write_imposters(impData, exp_num, act, us, featuresType)