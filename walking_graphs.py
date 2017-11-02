import glob
from sklearn.model_selection import train_test_split  
from sklearn import svm
import numpy as np
import pandas
import matplotlib.pyplot as plt


filenames = glob.glob('./AutoEncoderMyData/AEResult_featuresFilt_Running*.csv')
counter = 1
                      
for item in filenames:
    # Load current dataset
    url = item
    data = pandas.read_csv(url, header = None, engine='python')
    plt.plot(data, c='b', lw=1.5)
    plt.title('Running user ' + str(counter))
    plt.show()
    counter += 1