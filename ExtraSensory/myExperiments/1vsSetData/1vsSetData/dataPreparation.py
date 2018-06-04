import numpy as np
from sklearn.datasets import dump_svmlight_file
import os
import pandas as pd
import random
import math

from random import randint

activities = ["running", "walkingDownstairs", "walkingUpstairs"]

outliers_class_numbers = [0]

for act in activities:
    
    for neg_classes in outliers_class_numbers:
        allUsers = np.arange(1,57)
        random_users = []   
        for iter in range(1,6):
            randUs = random.choice(allUsers)
            random_users.append(randUs)
            index = np.argwhere(allUsers==randUs)
            allUsers = np.delete(allUsers, index)

        for us in range(1,57):
            allUsers = np.arange(1,57)
            
            allUsers = np.delete(allUsers, np.argwhere(allUsers==us))
            filepath = '../extractedFeatures/' + act + '/featuresFilt_' + act + '#' + str(us) + '.csv'
            if os.path.isfile(filepath):
                totalData = pd.read_csv(filepath)
                totalData.drop(["user"], axis=1, inplace=True)
                totalData = np.asarray(totalData, dtype= np.float32)

                splitted = np.array_split(totalData, 2)
                toTrain = splitted[0]
                y_train = np.ones(toTrain.shape[0])
                toTest = splitted[1]
                y_test = np.ones(toTest.shape[0]).astype('int')

                for i in range(neg_classes):
                    randUs = random.choice(allUsers)
                    index = np.argwhere(allUsers==randUs)
                    allUsers = np.delete(allUsers, index)
                    filepath = '../extractedFeatures/' + act + '/featuresFilt_' + act + '#' + str(randUs) + '.csv'
                    data = pd.read_csv(filepath)
                    data.drop(["user"], axis=1, inplace=True)
                    data = np.asarray(data, dtype= np.float32)
                    
                    data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0]*0.3)), :]
                    splitted = np.array_split(data, 3)
                    toTrain = np.vstack([toTrain, splitted[0]])

                    y_data = np.empty(splitted[0].shape[0])
                    y_data.fill(-1)
                    y_data = y_data.astype('int')
                    y_train = np.concatenate([y_train, y_data])

                    toTest = np.vstack([toTest, splitted[1]])
                    toTest = np.vstack([toTest, splitted[2]])

                    y_data = np.empty(splitted[1].shape[0])
                    y_data.fill(-1)
                    y_data = y_data.astype('int')
                    y_test = np.concatenate([y_test, y_data])

                    y_data = np.empty(splitted[2].shape[0])
                    y_data.fill(-1)
                    y_data = y_data.astype('int')
                    y_test = np.concatenate([y_test, y_data])

                for restUs in allUsers:
                    filepath = '../extractedFeatures/' + act + '/featuresFilt_' + act + '#' + str(restUs) + '.csv'
                    if os.path.isfile(filepath):
                        data = pd.read_csv(filepath)
                        data.drop(["user"], axis=1, inplace=True)
                        data = np.asarray(data, dtype= np.float32)
                        
                        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0]*0.2)), :]

                        y_data = np.empty(data.shape[0])
                        y_data.fill(-1)
                        y_data = y_data.astype('int')

                        toTest = np.vstack([toTest, data])
                        y_test = np.concatenate([y_test, y_data])


                if not os.path.isdir(act):
                    os.makedirs(act) 
                if not os.path.isdir(act + '/negClasses' + str(neg_classes)):
                    os.makedirs(act + '/negClasses' + str(neg_classes))    
                dump_svmlight_file(toTrain, y_train, act + '/negClasses' + str(neg_classes) + '/train#' + str(us) + '.data', zero_based=True)
                dump_svmlight_file(toTest, y_test, act + '/negClasses' + str(neg_classes) + '/test#' + str(us) + '.data', zero_based=True)