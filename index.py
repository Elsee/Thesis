# Load libraries
import pandas
import matplotlib.pyplot as plt
from sklearn import preprocessing
from statsmodels.tsa.stattools import pacf
import numpy as np
import math
import statsmodels.tsa.ar_model as ar_model
import statsmodels.tsa.arima_model as arima_model
from scipy.fftpack import fft
import pywt
from sklearn.model_selection import train_test_split

def showPlot(data):
    data.plot(subplots=True)
    plt.show()
    
def smoothActivityData(ds):
    #Filter data by applying moving average filter of order 3
    smoothed = dataset.rolling(window=3, min_periods=1).mean()
 #   smoothed.to_csv('D:/Study/Thesis/System/myProcessedData/movingAvg'+str(userNum)+'.csv', index = False)
    return smoothed

def windows(d, w, t, dsType):
    r = np.arange(len(d))
    s = r[::t]
    tSamples = [];
    for i in s:
        curRange = d.iloc[i:i+w]
        curRange = curRange.assign(user=userNum)
     #   curRange.to_csv('D:/Study/Thesis/System/myTrainingData/' + activityType + '/User' + str(userNum) + '/' + dsType + '/train' + str(i) + '.csv', index = False)
        tSamples.append(curRange)
    return tSamples

def normalization(data):
    return preprocessing.normalize(data).tolist()

#Function for IQR
#from sklearn.preprocessing import StandardScaler
def IQRcomp(samples):
    #IQRs = pandas.DataFrame(columns = {'x', 'y', 'z'})
    IQRs = []
    for sample in samples:
        # Computing IQR
        Q1x = sample['accx'].quantile(0.25)
        Q3x = sample['accx'].quantile(0.75)
        IQRx = Q1x - Q3x
        Q1y = sample['accy'].quantile(0.25)
        Q3y = sample['accy'].quantile(0.75)
        IQRy = Q1y - Q3y
        Q1z = sample['accz'].quantile(0.25)
        Q3z = sample['accz'].quantile(0.75)
        IQRz = Q1z - Q3z 
        IQRs.append([IQRx, IQRy, IQRz])  
    IQRs = normalization(IQRs)
    return IQRs

#Function for autocorrelation
def autoCorComp(samples):
    autoCors = []
    
    for sample in samples:
        autoCorx = sample['accx'].autocorr(lag=1)
        autoCory = sample['accy'].autocorr(lag=1)
        autoCorz = sample['accz'].autocorr(lag=1)
        autoCors.append([autoCorx, autoCory, autoCorz])
    autoCors = normalization(autoCors)
    return autoCors
       
#Function for partial autocorrelation
def pAutoCorComp(samples): 
    pAutoCors = []
    
    for sample in samples:
        pAutoCorx = pacf(np.asarray(sample['accx']), nlags=1, method='ywm')[1]
        pAutoCory = pacf(np.asarray(sample['accy']), nlags=1, method='ywm')[1]
        pAutoCorz = pacf(np.asarray(sample['accz']), nlags=1, method='ywm')[1]
        pAutoCors.append([pAutoCorx, pAutoCory, pAutoCorz])
    pAutoCors = normalization(pAutoCors)
    return pAutoCors

#Function for mean
def meanComp(samples):
    means = []
    
    for sample in samples:
        meanx = sample['accx'].mean()
        meany = sample['accy'].mean()
        meanz = sample['accz'].mean()
        means.append([meanx, meany, meanz])
    means = normalization(means)
    return means

#Function for median
def medianComp(samples):
    medians = []
    
    for sample in samples:
        medianx = sample['accx'].median()
        mediany = sample['accy'].median()
        medianz = sample['accz'].median()
        medians.append([medianx, mediany, medianz])
    medians = normalization(medians)
    return medians

#Function for variance
def varComp(samples):
    variances = []
    
    for sample in samples:
        varx = sample['accx'].var()
        vary = sample['accy'].var()
        varz = sample['accz'].var()
        variances.append([varx, vary, varz])
    variances = normalization(variances)
    return variances

#Function for std dev
def stdDevComp(samples):
    stdDevs = []
    
    for sample in samples:
        stdDevx = sample['accx'].std()
        stdDevy = sample['accy'].std()
        stdDevz = sample['accz'].std()
        stdDevs.append([stdDevx, stdDevy, stdDevz])
    stdDevs = normalization(stdDevs)
    return stdDevs

def arCoefComp(samples):
    arCoefs = []
    
    for sample in samples:
        ARmodX = ar_model.AR(np.asarray(sample['accx']))
        arCoefx = (ARmodX.fit(maxlag=1, trend='nc')).params
        ARmodY = ar_model.AR(np.asarray(sample['accy']))
        arCoefy = (ARmodY.fit(maxlag=1, trend='nc')).params
        ARmodZ = ar_model.AR(np.asarray(sample['accz']))
        arCoefz = (ARmodZ.fit(maxlag=1, trend='nc')).params
        arCoefs.append([arCoefx, arCoefy, arCoefz])
    return arCoefs

def armaCoefComp(samples):
    armaCoefs = []
    
    for sample in samples:
        armaCoefx = ((arima_model.ARMA(np.asarray(sample['accx']), order =(0,0)).fit(maxlag=1, trend='nc')).params)[1]
        armaCoefy = ((arima_model.ARMA(np.asarray(sample['accy']), order =(0,0)).fit(maxlag=1, trend='nc')).params)[1]
        armaCoefz = ((arima_model.ARMA(np.asarray(sample['accz']), order =(0,0)).fit(maxlag=1, trend='nc')).params)[1]
        armaCoefs.append([armaCoefx, armaCoefy, armaCoefz])
    return armaCoefs

def fftComp(samples):
    ffts = []
    
    for sample in samples: 
        fftx = fft(sample['accx'])
        Afx = np.amax(np.abs(fftx))
        ffty= fft(sample['accy'])
        Afy = np.amax(np.abs(ffty))
        fftz = fft(sample['accz'])
        Afz = np.amax(np.abs(fftz))
        ffts.append([Afx, Afy, Afz])
    return ffts

def pywtComp(samples):
    pywts = []
    
    for sample in samples:
        coefx, freqsx = pywt.cwt(sample['accx'],np.arange(1,31),'gaus1')
        coefy, freqsy = pywt.cwt(sample['accy'],np.arange(1,31),'gaus1')
        coefz, freqsz = pywt.cwt(sample['accz'],np.arange(1,31),'gaus1')
        mCoefx = coefx.mean()
        mCoefy = coefy.mean()
        mCoefz = coefz.mean()
        pywts.append([mCoefx, mCoefy, mCoefz])
    return pywts

userNum = 1
activityType = 'Jogging'
names = ['accx', 'accy', 'accz']
dataset = pandas.read_csv('D:/Study/Thesis/System/myProcessedData/total' + activityType + str(userNum) + '.csv', header = 0, names=names)

#showPlot(dataset)

smoothed = smoothActivityData(dataset)

step = int(math.floor(50 * 0.7))

def FFT(dataset):
    FFTArr = []
    for item in dataset: 
        FFTArr.append(pandas.DataFrame(fft(item), columns = {'accx', 'accy', 'accz', 'user'}))
    return FFTArr

def ARmodel(dataset):
    arCoefs = []
    
    for item in dataset:
        arCoefs.append(pandas.DataFrame(ar_model.AR(item), columns = {'accx', 'accy', 'accz', 'user'}))
    return arCoefs

#Training Samples
tSamplesOrig = windows(dataset, 50, step, 'original')   
tSamplesFilt = windows(smoothed, 50, step, 'filtered')
FFTOrig = FFT(tSamplesOrig)
FFTFilt = FFT(tSamplesFilt)

tFeaturesOrig = pandas.DataFrame()
tFeaturesFilt = tFeaturesOrig;

tFeaturesOrig['IQR'] = IQRcomp(tSamplesOrig)
tFeaturesFilt['IQR'] = IQRcomp(tSamplesFilt)
tFeaturesOrig['autocorrelation'] = autoCorComp(tSamplesOrig)
tFeaturesFilt['autocorrelation'] = autoCorComp(tSamplesFilt)
tFeaturesOrig['partial autocorrelation'] = pAutoCorComp(tSamplesOrig)
tFeaturesFilt['partial autocorrelation'] = pAutoCorComp(tSamplesFilt)
tFeaturesOrig['mean'] = meanComp(tSamplesOrig)
tFeaturesFilt['mean'] = meanComp(tSamplesFilt)
tFeaturesOrig['median'] = medianComp(tSamplesOrig)
tFeaturesFilt['median'] = medianComp(tSamplesFilt)
tFeaturesOrig['variance'] = varComp(tSamplesOrig)
tFeaturesFilt['variance'] = varComp(tSamplesFilt)
tFeaturesOrig['standard deviation'] = stdDevComp(tSamplesOrig)
tFeaturesFilt['standard deviation'] = stdDevComp(tSamplesFilt)
tFeaturesOrig['user'] = 1
tFeaturesFilt['user'] = 1
#tFeaturesOrig['coefficient of autoregressive (AR) model'] = arCoefComp(tSamplesOrig)
#tFeaturesFilt['coefficient of autoregressive (AR) model'] = arCoefComp(tSamplesFilt)
#tFeaturesOrig['coefficient of autoregressive-moving-average (ARMA) model'] = armaCoefComp(tSamplesOrig)
#tFeaturesFilt['coefficient of autoregressive-moving-average (ARMA) model'] = armaCoefComp(tSamplesFilt)

#tFeaturesOrig['FFT IQR'] = IQRcomp(FFTOrig)
#tFeaturesFilt['FFT IQR'] = IQRcomp(FFTFilt)
#tFeaturesOrig['FFT autocorrelation'] = autoCorComp(FFTOrig)
#tFeaturesFilt['FFT autocorrelation'] = autoCorComp(FFTFilt)
#tFeaturesOrig['FFT partial autocorrelation'] = pAutoCorComp(FFTOrig)
#tFeaturesFilt['FFT partial autocorrelation'] = pAutoCorComp(FFTFilt)
#tFeaturesOrig['FFT mean'] = meanComp(FFTOrig)
#tFeaturesFilt['FFT mean'] = meanComp(FFTFilt)
#tFeaturesOrig['FFT median'] = medianComp(FFTOrig)
#tFeaturesFilt['FFT median'] = medianComp(FFTFilt)
#tFeaturesOrig['FFT variance'] = varComp(FFTOrig)
#tFeaturesFilt['FFT variance'] = varComp(FFTFilt)
#tFeaturesOrig['FFT standard deviation'] = stdDevComp(FFTOrig)
#tFeaturesFilt['FFT standard deviation'] = stdDevComp(FFTFilt)

#tFeaturesOrig['wavelet coefficient'] = pywtComp(tSamplesOrig)
#tFeaturesFilt['wavelet coefficient'] = pywtComp(tSamplesFilt)



