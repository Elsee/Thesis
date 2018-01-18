from keras.layers import Input, Dense
from keras.models import Model

import keras.backend as K

def contractive_autoencoder(encoding_layer_dim, input_shape, X, X_test, lam=1e-3):
    N = input_shape
    N_hidden = encoding_layer_dim

    inputs = Input(shape=(N,))
    encoded = Dense(N_hidden, activation='sigmoid', name='encoded')(inputs)
    outputs = Dense(N, activation='linear')(encoded)

    model = Model(inputs, outputs)
    
    encoder = Model(inputs, encoded)

    def contractive_loss(y_pred, y_true):
        mse = K.mean(K.square(y_true - y_pred), axis=1)

        W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = model.get_layer('encoded').output
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

        return mse + contractive
        
    model.compile(optimizer='adam', loss=contractive_loss, metrics=['accuracy'])
    model.fit(X, X, 
              batch_size=32, 
              nb_epoch=350)

    return model, encoder


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

users = [1,2,3,4,5,6]
activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]
features =  ["featuresOrig", "featuresFilt"]

for feature in features:

    for act in activities:
        
        for us in users:
            totalData = pd.read_csv('../../myTrainingData/' + feature + '_' + act + '#' + str(us) + '.csv');
            totalData.drop(["user"], axis=1, inplace=True)
            totalData = sc.fit_transform(np.asarray(totalData, dtype= np.float32));
            
# =============================================================================
#             UNCOMMENT FOR STATISTICAL+WAVELET/TIME+FFT DATA
# =============================================================================
#            statisticalWaveData = np.concatenate((totalData[:,0:12], totalData[:,18:27], totalData[:, 36:57]), axis=1)
#            TimeFFTData = np.concatenate((totalData[:,27:36], totalData[:, 12:18]), axis=1)
#            
#            x_train_stat_wave, x_test_stat_wave = train_test_split(statisticalWaveData, test_size=0.2)
#            x_train_time_fft, x_test_time_fft = train_test_split(TimeFFTData, test_size=0.2)
#            
#            autoencoder_stat_wave, encoder_stat_wave = contractive_autoencoder(21, 42, x_train_stat_wave, x_test_stat_wave);
#            
#            autoencoder_time_fft, encoder_time_fft = contractive_autoencoder(8, 15, x_train_time_fft, x_test_time_fft);
#            
#            encoded_stats_wave = encoder_stat_wave.predict(statisticalWaveData)
#            encoded_time_fft = encoder_time_fft.predict(TimeFFTData)
#            
#            concat_encoded = np.concatenate((encoded_stats_wave, encoded_time_fft), axis=1)
#            
#            x_train_fused, x_test_fused = train_test_split(concat_encoded, test_size=0.2)
#            
#            autoencoder_fused, encoder_fused = contractive_autoencoder(15, 29, x_train_fused, x_test_fused);
#            
#            encoded_fused = encoder_fused.predict(concat_encoded)
#
#            np.savetxt("./resultsFusedContractive3AEStatWavelet/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')
                       
# =============================================================================
#            UNCOMMENT FOR STATISTICAL+TIME/FFT+WAVELET DATA 
# =============================================================================
#        
#            statisticalTimeData = np.concatenate((totalData[:,0:12], totalData[:,18:36]), axis=1)
#            fftWaveletData = np.concatenate((totalData[:, 12:18], totalData[:, 36:57]), axis=1)
#            
#            x_train_stat_time, x_test_stat_time = train_test_split(statisticalTimeData, test_size=0.2)
#            x_train_fft_wavelet, x_test_fft_wavelet = train_test_split(fftWaveletData, test_size=0.2)
#            
#            autoencoder_stat_time, encoder_stat_time = contractive_autoencoder(16, 30,x_train_stat_time, x_test_stat_time);
#            autoencoder_fft_wavelet, encoder_fft_wavelet = contractive_autoencoder(16, 27, x_train_fft_wavelet, x_test_fft_wavelet);
#            
#            encoded_stats_wave = encoder_stat_time.predict(statisticalTimeData)
#            encoded_fft_wavelet = encoder_fft_wavelet.predict(fftWaveletData)
#            
#            concat_encoded = np.concatenate((encoded_stats_wave, encoded_fft_wavelet), axis=1)
#            
#            x_train_fused, x_test_fused = train_test_split(concat_encoded, test_size=0.2)
#            
#            autoencoder_fused, encoder_fused = contractive_autoencoder(16, 32, x_train_fused, x_test_fused);
#            encoded_fused = encoder_fused.predict(concat_encoded)
#
#            np.savetxt("./resultsFusedContractive3AEStatTime/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')


# =============================================================================
#            UNCOMMENT FOR STATISTICAL+FFT/TIME+WAVELET DATA 
# =============================================================================
        
            statisticalFFTData = totalData[:,0:27]
            TimeWaveletData = totalData[:,27:57]
            
            x_train_stat_fft, x_test_stat_fft = train_test_split(statisticalFFTData, test_size=0.2)
            x_train_time_wavelet, x_test_time_wavelet = train_test_split(TimeWaveletData, test_size=0.2)
            
            autoencoder_stat_fft, encoder_stat_fft = contractive_autoencoder(14, 27, x_train_stat_fft, x_test_stat_fft);
            autoencoder_time_wavelet, encoder_time_wavelet = contractive_autoencoder(15, 30, x_train_time_wavelet, x_test_time_wavelet);
            
            encoded_stats_fft = encoder_stat_fft.predict(statisticalFFTData)
            encoded_time_wavelet = encoder_time_wavelet.predict(TimeWaveletData)
            
            concat_encoded = np.concatenate((encoded_stats_fft, encoded_time_wavelet), axis=1)
            
            x_train_fused, x_test_fused = train_test_split(concat_encoded, test_size=0.2)
            
            autoencoder_fused, encoder_fused = contractive_autoencoder(15, 29, x_train_fused, x_test_fused);
            encoded_fused = encoder_fused.predict(concat_encoded)

            np.savetxt("./resultsFusedContractive3AEStatFFT/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')