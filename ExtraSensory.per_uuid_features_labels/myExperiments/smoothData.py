# Load libraries
import pandas
import os;

def smoothAct(uN):
    userNum = uN
    
    #Filter data by applying moving average filter of order 3
    if os.path.isfile('./concatenatedData/walkingUpstairs/total_walkingUpstairs#' + str(userNum) + '.csv'):
        origDataset = pandas.read_csv('./concatenatedData/walkingUpstairs/total_walkingUpstairs#' + str(userNum) + '.csv', header=None)
        smoothed = origDataset.rolling(window=3, min_periods=1).mean()
        smoothed.to_csv('./filteredData/walkingUpstairs/movingAvg_walkingUpstairs#' +  str(userNum)+'.csv', index = False)
    

for us in range(56):
    smoothAct(us+1)
            