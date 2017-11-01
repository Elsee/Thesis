from sklearn.decomposition import PCA
import pandas

users = [1,2,3,4,5,6]
activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]
features =  ["featuresOrig", "featuresFilt"]
PCAcompNum = 40

for feature in features:
    for act in activities:
        for us in users:

            pca = PCA(n_components=PCAcompNum)
            
            processingFeatures = pandas.read_csv('./myTrainingData/' + feature +'_' + act + '#' + str(us) + '.csv', header = 0)
            processingFeatures.drop(["user"], axis=1, inplace=True)
                        
            XPCAreduced = pca.fit_transform(processingFeatures)
            XPCAreduced = pandas.DataFrame(XPCAreduced)
            XPCAreduced['user'] = us
            
            XPCAreduced.to_csv('./myTrainingData/' + feature + 'PCA' + str(PCAcompNum) + '_' + act + '#' + str(us) + '.csv', index = False)
            