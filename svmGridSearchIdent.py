# Grid Search

# Importing the libraries
import pandas
import glob
from sklearn.model_selection import train_test_split  
from sklearn import svm

def gridSearchSVM(feat, act):
    # Importing the dataset
    featureType = feat
    activityType = act

    filenames = glob.glob('./myTrainingData/' + featureType +'_' + activityType + '#*.csv')
    dataset = pandas.DataFrame()
    
    for item in filenames:
            # Load current dataset
            url = item
            data = pandas.read_csv(url, header = 0, engine='python')
            dataset = pandas.concat([dataset, data], ignore_index=True)
        
    y = dataset['user']
    dataset.drop(["user"], axis=1, inplace=True)
    X = dataset
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)   
    
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
        
    model = svm.SVC(kernel='rbf', random_state = 0)  
    model.fit(X_train, y_train)
    
    # Predicting the Test set results
#    preds = model.predict(test_data)
    
    # Making the Confusion Matrix
#    from sklearn.metrics import confusion_matrix
#    cm = confusion_matrix(test_target, preds)
#    
    # Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10, scoring="accuracy")
    accuracies.mean()
    accuracies.std()
    
    # Applying Grid Search to find the best model and the best parameters
    from sklearn.model_selection import GridSearchCV
    parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                  {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 'auto']}]
    grid_search = GridSearchCV(estimator = model,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10)
    grid_search = grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    

    with open('./gridSearchSVMresultsIdent.txt','a') as f:
        f.write('Best parameters for ' + featureType + ' of users doing activity ' + activityType + ': ' + str(best_parameters) + '. Best accuracy: ' + str(best_accuracy) + '\n')

activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]
features =  ["featuresFilt"]

for feature in features:
    for act in activities:
            gridSearchSVM(feature, act)