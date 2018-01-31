import glob
from sklearn.model_selection import train_test_split  
from sklearn import svm
import numpy as np
import pandas

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle

activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]
models = ["withoutAE", "results3AE", "results3AEDeep", "results5AE", "results5AEDeep"]
models_titles = ["without AE", "k=2 SAEs", "k=2 DAEs", "k=4 SAEs", "k=4 DAEs"]

fpr = dict()
tpr = dict()
roc_auc = dict()
thresholds = dict()

us = 5

for act in activities:
    
    activityType = act
 
    for model_type in models:
        if model_type == "withoutAE":
            filenames = glob.glob('../myTrainingData/featuresOrig_' + activityType + '#*.csv')
        else: 
            filenames = glob.glob('../AutoEncoderMyData/' + model_type + '/AEResult_featuresOrig_' + activityType + '#*.csv')                          
                                  
        del filenames[us-1]
    
        allUsersFeatures = pandas.DataFrame()
    
        for item in filenames:
            # Load current dataset
            url = item
            if model_type == "withoutAE":
                dataset = pandas.read_csv(url, skiprows=1, header = None, engine='python')
            else:
                dataset = pandas.read_csv(url, header = None, engine='python')
                
            allUsersFeatures = pandas.concat([allUsersFeatures, dataset], ignore_index=True)
    
        allUsersFeatures["target"] = -1
        impostors = allUsersFeatures["target"]
        allUsersFeatures.drop(["target"], axis=1, inplace=True)
    
        if model_type == "withoutAE":
            currentUserData = pandas.read_csv('../myTrainingData/featuresOrig_' + activityType + '#5.csv', skiprows=1, header = None)
        else:
            currentUserData = pandas.read_csv('../AutoEncoderMyData/' + model_type + '/AEResult_featuresOrig_' + activityType + '#5.csv', header = None)                          
            
        currentUserData['target'] = 1
        
        curUserTarget = currentUserData['target']
        
        currentUserData.drop(["target"], axis=1, inplace=True)
        
        train_data, test_data, train_target, test_target = train_test_split(currentUserData, curUserTarget, train_size = 0.8, test_size = 0.2)  
        
        if model_type == "results3AE":
            model = svm.OneClassSVM(nu=0.1, kernel='rbf')
        else:
            model = svm.OneClassSVM(nu=0.1, kernel='linear')  
        
        test_data_with_impostors = np.r_[test_data, allUsersFeatures]
        y_score = model.fit(train_data).decision_function(test_data_with_impostors) 
        
        y_pred_train =  model.predict(train_data) 
        y_pred_test = model.predict(test_data)
        
        y_labels = np.array([1]*test_data.shape[0]+[-1]*allUsersFeatures.shape[0])
        
        fpr[model_type], tpr[model_type], thresholds[model_type] = roc_curve(y_labels, y_score)
        roc_auc[model_type] = auc(fpr[model_type], tpr[model_type])
    
    feature_type = "featuresOrig"
    
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'lime', 'indigo'])
    idx = 0
    for i, color in zip(models, colors):
        plt.plot(fpr[i], tpr[i], color=color, label='{0} (area = {1:0.2f})'.format(models_titles[idx], roc_auc[i]))
        idx = idx + 1 
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.rc('axes', titlesize=15)     # fontsize of the axes title
    plt.rc('axes', labelsize=15)
    plt.rc('legend', fontsize=15) 
    plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=13)
    #        plt.title('ROCs for {0} obtained from {1} features'.format(act, feature_type))
    leg = plt.legend(loc="lower right")
    #leg.get_frame().set_alpha(0.5)
    plt.tight_layout()
    plt.savefig('./rocs_models/' + activityType + '_result.jpg', format='jpg')
    plt.close()

            
            