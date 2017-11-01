# Load libraries
import pandas

def smoothAct(uN, acT):
    userNum = uN
    activityType = acT
    names = ['accx', 'accy', 'accz']
    
    #Filter data by applying moving average filter of order 3
    origDataset = pandas.read_csv('./myProcessedData/total_' + activityType + '#' + str(userNum) + '.csv', header = 0, names=names)
    smoothed = origDataset.rolling(window=3, min_periods=1).mean()
    smoothed.to_csv('./myProcessedData/movingAvg_' + activityType + '#' +  str(userNum)+'.csv', index = False)

users = [1,2,3,4,5,6]
activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]

for act in activities:
    for us in users:
        smoothAct(us, act)