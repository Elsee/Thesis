# Grid Search

# Importing the libraries
import pandas
from sklearn.model_selection import train_test_split  
from sklearn import svm

def gridSearchSVM(feat, act, us):
    # Importing the dataset
    featureType = feat
    activityType = act
    userNum = us
    
    currentUserData = pandas.read_csv('./resultsFusedVariational3AEStatFFT/AEResult_' + featureType + '_' + activityType + '#' + str(userNum) + '.csv', header = 0)
    currentUserData['target'] = 1
    
    curUserTarget = currentUserData['target']
    
    currentUserData.drop(["target"], axis=1, inplace=True)
    
    train_data, test_data, train_target, test_target = train_test_split(currentUserData, curUserTarget, train_size = 0.8, test_size = 0.2)  
    
    model = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.6)  
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
    parameters = [{'nu': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99], 'kernel': ['linear']},
                  {'nu': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 'auto']}]
    grid_search = GridSearchCV(estimator = model,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10)
    grid_search = grid_search.fit(train_data, train_target)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    

    with open('./gridSearchRes/gridSearchSVMresults_3AEStatFFT.txt','a') as f:
        f.write('Best parameters for ' + featureType + ' of user number ' + str(userNum) + ' doing activity ' + activityType + ': ' + str(best_parameters) + '. Best accuracy: ' + str(best_accuracy) + '\n')


users = [1,2,3,4,5,6]
activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]
features =  ["featuresFilt"]

for feature in features:
    for act in activities:
        for us in users:
            gridSearchSVM(feature, act, us)