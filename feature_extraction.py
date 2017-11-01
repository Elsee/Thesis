# Load libraries
import pandas
import matplotlib.pyplot as plt
from sklearn import preprocessing
from statsmodels.tsa.stattools import pacf
import numpy as np
import math
import statsmodels.tsa.ar_model as ar_model
import statsmodels.tsa.statespace.sarimax as sarimax_model
from scipy.fftpack import fft
import pywt

def extractFatures(uN, aT):
    def showPlot(data):
        data.plot(subplots=True)
        plt.show()
    
    def windows(d, w, t, dsType):
        r = np.arange(len(d))
        s = r[::t]
        tSamples = [];
        for i in s:
            curRange = d.iloc[i:i+w]
            if (len(curRange) < 50):
                continue
            curRange = curRange.assign(user=userNum)
            tSamples.append(curRange)
        return tSamples
    
    userNum = uN
    activityType = aT
    names = ['accx', 'accy', 'accz']
    dataset = pandas.read_csv('./myProcessedData/total_' + activityType + '#' + str(userNum) + '.csv', header = 0, names=names)
    
    #showPlot(dataset)
    
    smoothed = pandas.read_csv('./myProcessedData/movingAvg_' + activityType + '#' + str(userNum) + '.csv', header = 0, names=names)
    
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
    
    #Function for IQR
    #from sklearn.preprocessing import StandardScaler
    def IQRcomp(samples):
        IQRs = pandas.DataFrame(columns = {'IQR_x', 'IQR_y', 'IQR_z'})
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
            tempIQRs = pandas.DataFrame([[IQRx, IQRy, IQRz]], columns = {'IQR_x', 'IQR_y', 'IQR_z'})
            IQRs = IQRs.append(tempIQRs, ignore_index=True)  
        return IQRs
    
    tFeaturesOrig = tFeaturesOrig.join(IQRcomp(tSamplesOrig), how='outer')
    tFeaturesFilt = tFeaturesFilt.join(IQRcomp(tSamplesFilt), how='outer')
    
    #Function for autocorrelation
    def autoCorComp(samples):
        autoCors = pandas.DataFrame(columns = {'AutoCor_x', 'AutoCor_y', 'AutoCor_z'})
        
        for sample in samples:
            autoCorx = sample['accx'].autocorr(lag=1)
            autoCory = sample['accy'].autocorr(lag=1)
            autoCorz = sample['accz'].autocorr(lag=1)
            tempAutoCors  = pandas.DataFrame([[autoCorx, autoCory, autoCorz]], columns = {'AutoCor_x', 'AutoCor_y', 'AutoCor_z'})
            autoCors = autoCors.append(tempAutoCors, ignore_index=True)
        return autoCors
    
    tFeaturesOrig = tFeaturesOrig.join(autoCorComp(tSamplesOrig), how='outer')
    tFeaturesFilt = tFeaturesFilt.join(autoCorComp(tSamplesFilt), how='outer')   
         
    #Function for partial autocorrelation
    def pAutoCorComp(samples): 
        pAutoCors = pandas.DataFrame(columns = {'PartAutoCor_x', 'PartAutoCor_y', 'PartAutoCor_z'})
        
        for sample in samples:
            pAutoCorx = pacf(np.asarray(sample['accx']), nlags=1, method='ywm')[1]
            pAutoCory = pacf(np.asarray(sample['accy']), nlags=1, method='ywm')[1]
            pAutoCorz = pacf(np.asarray(sample['accz']), nlags=1, method='ywm')[1]
            tempPartAutoCors  = pandas.DataFrame([[pAutoCorx, pAutoCory, pAutoCorz]], columns = {'PartAutoCor_x', 'PartAutoCor_y', 'PartAutoCor_z'})
            pAutoCors = pAutoCors.append(tempPartAutoCors, ignore_index=True)
        return pAutoCors
    
    tFeaturesOrig = tFeaturesOrig.join(pAutoCorComp(tSamplesOrig), how='outer')
    tFeaturesFilt = tFeaturesFilt.join(pAutoCorComp(tSamplesFilt), how='outer')
    
    #Function for mean
    def meanComp(samples):
        means = pandas.DataFrame(columns = {'mean_x', 'mean_y', 'mean_z'})
        
        for sample in samples:
            meanx = sample['accx'].mean()
            meany = sample['accy'].mean()
            meanz = sample['accz'].mean()
            tempMeans  = pandas.DataFrame([[meanx, meany, meanz]], columns = {'mean_x', 'mean_y', 'mean_z'})
            means = means.append(tempMeans, ignore_index=True)
        return means
    
    tFeaturesOrig = tFeaturesOrig.join(meanComp(tSamplesOrig), how='outer')
    tFeaturesFilt = tFeaturesFilt.join(meanComp(tSamplesFilt), how='outer')
    
    #Function for Energy
    def energyComp(samples):
        energies = pandas.DataFrame(columns = {'energy_x', 'energy_y', 'energy_z'})
        
        for sample in samples:
            energyx = sum(np.power(abs(sample['accx']),2))/(len(sample))
            energyy = sum(np.power(abs(sample['accy']),2))/(len(sample))
            energyz = sum(np.power(abs(sample['accz']),2))/(len(sample))
            tempEnergies  = pandas.DataFrame([[energyx, energyy, energyz]], columns = {'energy_x', 'energy_y', 'energy_z'})
            energies = energies.append(tempEnergies, ignore_index=True)
        return energies
    
    tFeaturesOrig = tFeaturesOrig.join(energyComp(FFTOrig), how='outer')
    tFeaturesFilt = tFeaturesFilt.join(energyComp(FFTFilt), how='outer')
    
    #Function for Spectral Entropy
    def entropyComp(samples):
        entropies = pandas.DataFrame(columns = {'entropy_x', 'entropy_y', 'entropy_z'})
        
        for sample in samples:
            PSDx = np.power(abs(sample['accx']),2)/(len(sample))
            PSDxNorm = PSDx/sum(PSDx)
            entropyx = -sum(PSDxNorm*np.log(PSDxNorm))
            PSDy = np.power(abs(sample['accy']),2)/(len(sample))
            PSDyNorm = PSDy/sum(PSDy)
            entropyy = -sum(PSDyNorm*np.log(PSDyNorm))
            PSDz = np.power(abs(sample['accz']),2)/(len(sample))
            PSDzNorm = PSDz/sum(PSDz)
            entropyz = -sum(PSDzNorm*np.log(PSDzNorm))
            tempEntropies  = pandas.DataFrame([[entropyx, entropyy, entropyz]], columns = {'entropy_x', 'entropy_y', 'entropy_z'})
            entropies = entropies.append(tempEntropies, ignore_index=True)
        return entropies
    
    tFeaturesOrig = tFeaturesOrig.join(entropyComp(FFTOrig), how='outer')
    tFeaturesFilt = tFeaturesFilt.join(entropyComp(FFTFilt), how='outer')
    
    #Function for median
    def medianComp(samples):
        medians = pandas.DataFrame(columns = {'median_x', 'median_y', 'median_z'})
        
        for sample in samples:
            medianx = sample['accx'].median()
            mediany = sample['accy'].median()
            medianz = sample['accz'].median()
            tempMedians  = pandas.DataFrame([[medianx, mediany, medianz]], columns = {'median_x', 'median_y', 'median_z'})
            medians = medians.append(tempMedians, ignore_index=True)
        return medians
    
    tFeaturesOrig = tFeaturesOrig.join(medianComp(tSamplesOrig), how='outer')
    tFeaturesFilt = tFeaturesFilt.join(medianComp(tSamplesFilt), how='outer')
    
    #Function for variance
    def varComp(samples):
        variances = pandas.DataFrame(columns = {'variance_x', 'variance_y', 'variance_z'})
        
        for sample in samples:
            varx = sample['accx'].var()
            vary = sample['accy'].var()
            varz = sample['accz'].var()
            tempVariances  = pandas.DataFrame([[varx, vary, varz]], columns = {'variance_x', 'variance_y', 'variance_z'})
            variances = variances.append(tempVariances, ignore_index=True)
        return variances
    
    tFeaturesOrig = tFeaturesOrig.join(varComp(tSamplesOrig), how='outer')
    tFeaturesFilt = tFeaturesFilt.join(varComp(tSamplesFilt), how='outer')
    
    #Function for std dev
    def stdDevComp(samples):
        stdDevs = pandas.DataFrame(columns = {'stdDev_x', 'stdDev_y', 'stdDev_z'})
        
        for sample in samples:
            stdDevx = sample['accx'].std()
            stdDevy = sample['accy'].std()
            stdDevz = sample['accz'].std()
            tempStdDevs  = pandas.DataFrame([[stdDevx, stdDevy, stdDevz]], columns = {'stdDev_x', 'stdDev_y', 'stdDev_z'})
            stdDevs = stdDevs.append(tempStdDevs, ignore_index=True)
        return stdDevs
    
    tFeaturesOrig = tFeaturesOrig.join(stdDevComp(tSamplesOrig), how='outer')
    tFeaturesFilt = tFeaturesFilt.join(stdDevComp(tSamplesFilt), how='outer')
    
    def arCoefComp(samples):
        arCoefs = pandas.DataFrame(columns = {'ar_x1', 'ar_x2', 'ar_x3', 'ar_y1', 'ar_y2', 'ar_y3', 'ar_z1', 'ar_z2', 'ar_z3'})
        
        for sample in samples:
            ARmodX = ar_model.AR(np.asarray(sample['accx']))
            arCoefx = (ARmodX.fit(maxlag=3, trend='nc')).params
            ARmodY = ar_model.AR(np.asarray(sample['accy']))
            arCoefy = (ARmodY.fit(maxlag=3, trend='nc')).params
            ARmodZ = ar_model.AR(np.asarray(sample['accz']))
            arCoefz = (ARmodZ.fit(maxlag=3, trend='nc')).params
            tempArCoefs  = pandas.DataFrame([[arCoefx[0], arCoefx[1], arCoefx[2], arCoefy[0], arCoefy[1], arCoefy[2], arCoefz[0], arCoefz[1], arCoefz[2]]], columns = {'ar_x1', 'ar_x2', 'ar_x3', 'ar_y1', 'ar_y2', 'ar_y3', 'ar_z1', 'ar_z2', 'ar_z3'})
            arCoefs = arCoefs.append(tempArCoefs, ignore_index=True)
        return arCoefs
    
    tFeaturesOrig = tFeaturesOrig.join(arCoefComp(tSamplesOrig), how='outer')
    tFeaturesFilt = tFeaturesFilt.join(arCoefComp(tSamplesFilt), how='outer')
    
#    def armaCoefComp(samples):
#        armaCoefs = pandas.DataFrame(columns = {'arma_x1', 'arma_x2', 'arma_x3', 'arma_x4', 'arma_y1', 'arma_y2', 'arma_y3', 'arma_y4', 'arma_z1', 'arma_z2', 'arma_z3', 'arma_z4'})
#        
#        for sample in samples:
#            armaCoefx = ((sarimax_model.SARIMAX(np.asarray(sample['accx']), order = (2,0,2), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)).params) 
#            armaCoefy = ((sarimax_model.SARIMAX(np.asarray(sample['accy']), order = (2,0,2), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)).params) 
#            armaCoefz = ((sarimax_model.SARIMAX(np.asarray(sample['accz']), order = (2,0,2), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)).params) 
#            tempArmaCoefs  = pandas.DataFrame([[armaCoefx[0], armaCoefx[1], armaCoefx[2], armaCoefx[3], armaCoefy[0], armaCoefy[1], armaCoefy[2], armaCoefy[3], armaCoefz[0], armaCoefz[1], armaCoefz[2], armaCoefz[3]]],columns = {'arma_x1', 'arma_x2', 'arma_x3', 'arma_x4', 'arma_y1', 'arma_y2', 'arma_y3', 'arma_y4', 'arma_z1', 'arma_z2', 'arma_z3', 'arma_z4'})
#            armaCoefs = armaCoefs.append(tempArmaCoefs, ignore_index=True)
#        return armaCoefs
#    
#    tFeaturesOrig = tFeaturesOrig.join(armaCoefComp(tSamplesOrig), how='outer')  
#    tFeaturesFilt = tFeaturesFilt.join(armaCoefComp(tSamplesFilt), how='outer') 
    
    #function for  wavelet coefficients sum, squared sum, energy
    def pywtComp(samples):
        pywts = pandas.DataFrame(columns = {'wtSumDet_x', 'wtSumDet_y', 'wtSumDet_z','wtSumApr_x', 'wtSumApr_y', 'wtSumApr_z', 'squaredwtSumDet_x', 'squaredwtSumDet_y', 'squaredwtSumDet_z', 'squaredwtSumApr_x', 'squaredwtSumApr_y', 'squaredwtSumApr_z','enDetx', 'enApprx', 'enratiox', 'enDety', 'enAppry', 'enratioy', 'enDetz', 'enApprz', 'enratioz'})
        
        for sample in samples:
            cAx, cDx = pywt.dwt(sample['accx'], 'db1')
            cAy, cDy = pywt.dwt(sample['accy'], 'db1')
            cAz, cDz = pywt.dwt(sample['accz'], 'db1')
            Edetailx = math.sqrt(sum(cDx**2)/len(cDx))
            Eapproxx = math.sqrt(sum(cAx**2)/len(cDx))
            Eratiox = Edetailx/Eapproxx
            Edetaily = math.sqrt(sum(cDy**2)/len(cDy))
            Eapproxy = math.sqrt(sum(cAy**2)/len(cDy))
            Eratioy = Edetaily/Eapproxy
            Edetailz = math.sqrt(sum(cDz**2)/len(cDz))
            Eapproxz = math.sqrt(sum(cAz**2)/len(cDz))
            Eratioz = Edetailz/Eapproxz
            sumCoefxDet = sum(cDx)
            sumCoefyDet = sum(cDy)
            sumCoefzDet = sum(cDz)
            sumCoefxApr = sum(cAx)
            sumCoefyApr = sum(cAy)
            sumCoefzApr = sum(cAz)
            squaredSumCoefxDet = sumCoefxDet**2
            squaredSumCoefyDet = sumCoefyDet**2
            squaredSumCoefzDet = sumCoefzDet**2
            squaredSumCoefxApr = sumCoefxApr**2
            squaredSumCoefyApr = sumCoefyApr**2
            squaredSumCoefzApr = sumCoefzApr**2
            tempPywts  = pandas.DataFrame([[sumCoefxDet, sumCoefyDet, sumCoefzDet, sumCoefxApr, sumCoefyApr, sumCoefzApr, squaredSumCoefxDet, squaredSumCoefyDet, squaredSumCoefzDet, squaredSumCoefxApr, squaredSumCoefyApr, squaredSumCoefzApr, Edetailx, Eapproxx, Eratiox, Edetaily, Eapproxy, Eratioy, Edetailz, Eapproxz, Eratioz]], columns = {'wtSumDet_x', 'wtSumDet_y', 'wtSumDet_z','wtSumApr_x', 'wtSumApr_y', 'wtSumApr_z', 'squaredwtSumDet_x', 'squaredwtSumDet_y', 'squaredwtSumDet_z', 'squaredwtSumApr_x', 'squaredwtSumApr_y', 'squaredwtSumApr_z', 'enDetx', 'enApprx', 'enratiox', 'enDety', 'enAppry', 'enratioy', 'enDetz', 'enApprz', 'enratioz'})
            pywts = pywts.append(tempPywts, ignore_index=True)
        return pywts
    
    tFeaturesOrig = tFeaturesOrig.join(pywtComp(tSamplesOrig), how='outer')
    tFeaturesFilt = tFeaturesFilt.join(pywtComp(tSamplesFilt), how='outer')
        
    tFeaturesOrig['user'] = userNum
    tFeaturesFilt['user'] = userNum
    
    tFeaturesOrig.to_csv('./myTrainingData/featuresOrig_' + activityType + '#' + str(userNum) + '.csv', index = False)
    tFeaturesFilt.to_csv('./myTrainingData/featuresFilt_' + activityType + '#' + str(userNum) + '.csv', index = False)

users = [1,2,3,4,5,6]
activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"] 

for act in activities:
    for us in users:
        extractFatures(us, act)