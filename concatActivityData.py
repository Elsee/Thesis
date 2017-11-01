import pandas
import glob

def concatAct(uN, acT):
    activityType = acT
    userNum = uN
    
    filenames = glob.glob('./Dataset/User '+ str(userNum) + '/' + activityType + ',*.csv')
    
    totalDataset = pandas.DataFrame()
    
    for item in filenames:
        # Load current dataset
        url = item
        #choose only accelerometer data
        names = ['accx', 'accy', 'accz']
        dataset = pandas.read_csv(url, header = 0, names=names, usecols = [0,1,2], skiprows = 50, skipfooter = 51, engine='python')
        totalDataset = pandas.concat([totalDataset, dataset], ignore_index=True)
    
    totalDataset.to_csv('./myProcessedData/total_' + activityType + '#' + str(userNum) + '.csv', index = False)


users = [1,2,3,4,5,6]
activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]

for act in activities:
    for us in users:
        concatAct(us, act)