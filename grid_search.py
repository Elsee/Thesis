# Grid Search

# Importing the libraries
import pandas
import glob
from sklearn.model_selection import train_test_split  
from sklearn import svm
from time import gmtime, strftime

def gridSearchSVM(feat, act, us):
    # Importing the dataset
    featureType = feat
    activityType = act
    userNum = us
    
    filenames = glob.glob('./myTrainingData/' + featureType + '_' + activityType + '#*.csv')
    allUsersFeatures = pandas.DataFrame()
    
    for item in filenames:
        # Load current dataset
        url = item
        dataset = pandas.read_csv(url, header = 0, engine='python')
        allUsersFeatures = pandas.concat([allUsersFeatures, dataset], ignore_index=True)
    
    allUsersFeatures.drop(allUsersFeatures[allUsersFeatures.user == userNum].index, inplace=True)
    allUsersFeatures.drop(["user"], axis=1, inplace=True)
              
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    allUsersFeatures = sc.fit_transform(allUsersFeatures)
    
    currentUserData = pandas.read_csv('./myTrainingData/' + featureType + '_' + activityType + '#' + str(userNum) + '.csv', header = 0)
    currentUserData['target'] = 1
    
    curUserTarget = currentUserData['target']
    
    currentUserData.drop(["user", "target"], axis=1, inplace=True)
    
    currentUserData = sc.fit_transform(currentUserData)
    
    train_data, test_data, train_target, test_target = train_test_split(currentUserData, curUserTarget, train_size = 0.8, test_size = 0.2)  
    
    model = svm.OneClassSVM(kernel='rbf', random_state = 0)  
    model.fit(train_data) 
    
    # Predicting the Test set results
#    preds = model.predict(test_data)
    
    # Making the Confusion Matrix
#    from sklearn.metrics import confusion_matrix
#    cm = confusion_matrix(test_target, preds)
#    
    # Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = model, X = train_data, y = train_target, cv = 10, scoring="accuracy")
    accuracies.mean()
    accuracies.std()
    
    # Applying Grid Search to find the best model and the best parameters
    from sklearn.model_selection import GridSearchCV
    parameters = [{'nu': [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99], 'kernel': ['linear']},
                  {'nu': [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 'auto']}]
    grid_search = GridSearchCV(estimator = model,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10)
    grid_search = grid_search.fit(train_data, train_target)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    

    with open('./gridSearchSVMresults.txt','a') as f:
        f.write('Best parameters for ' + featureType + ' of user number ' + str(userNum) + ' doing activity ' + activityType + ': ' + str(best_parameters) + '. Best accuracy: ' + str(best_accuracy) + '\n')


users = [1,2,3,4,5,6]
activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]
features =  ["featuresFilt"]

with open('./gridSearchSVMresults.txt','a') as myfile:
                myfile.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n\n\n")

for feature in features:
    for act in activities:
        for us in users:
            gridSearchSVM(feature, act, us)