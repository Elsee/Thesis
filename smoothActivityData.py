# Load libraries
import pandas
import matplotlib.pyplot as plt

def smoothAct(uN, acT):
    userNum = uN
    activityType = acT
    names = ['accx', 'accy', 'accz']
    
    #Filter data by applying moving average filter of order 3
    origDataset = pandas.read_csv('./myProcessedData/total_' + activityType + '#' + str(userNum) + '.csv', header = 0, names=names)
    smoothed = origDataset.rolling(window=3, min_periods=1).mean()
    smoothed.to_csv('./myProcessedData/movingAvg_' + activityType + '#' +  str(userNum)+'.csv', index = False)
    plt.plot(smoothed['accx'], c='b', lw=1)
    plt.plot(smoothed['accy'], c='r', lw=1)
    plt.plot(smoothed['accz'], c='g', lw=1)
    plt.legend(['x-axis', 'y-axis', 'z-axis'], loc='upper left')
#    plt.title("User " + str(userNum) + " " + activityType)
    plt.tight_layout()
    plt.savefig("./Smoothed data results/"+activityType+str(userNum)+".jpg", format='jpg')
    plt.close()
    
users = [1,2,3,4,5,6]
activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]

for act in activities:
    for us in users:
        smoothAct(us, act)
            