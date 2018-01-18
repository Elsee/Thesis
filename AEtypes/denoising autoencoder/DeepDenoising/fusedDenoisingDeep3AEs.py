from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

def denoising_autoencoder(encoding_layer_dim, intermediate_dim1, intermediate_dim2, input_shape, X, X_noisy, X_test, X_test_noisy):
    # this is the size of our encoded representations
    encoding_dim = encoding_layer_dim
    # this is our input placeholder
    input_img = Input(shape=(input_shape,))
    # "encoded" is the encoded representation of the input
    h1 = Dense(intermediate_dim2, activation='relu')(input_img)
    h2 = Dense(intermediate_dim1, activation='relu')(h1)
    encoded = Dense(encoding_dim, activation='relu', name='encoded')(h2)
    # "decoded" is the lossy reconstruction of the input
    decoded_h2 = Dense(intermediate_dim1, activation='sigmoid')(encoded)
    decoded_h1 = Dense(intermediate_dim2, activation='sigmoid')(decoded_h2)
    decoded = Dense(input_shape, activation='sigmoid')(decoded_h1)
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)
    
    autoencoder.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    
    autoencoder.fit(X_noisy, X, 
              batch_size=32, 
              epochs=400,
              shuffle=True,
              validation_data=(X_test_noisy, X_test))
    
    return autoencoder, encoder

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
            totalData = pd.read_csv('../../../myTrainingData/' + feature + '_' + act + '#' + str(us) + '.csv');
            totalData.drop(["user"], axis=1, inplace=True)
            totalData = sc.fit_transform(np.asarray(totalData, dtype= np.float32));
            
# =============================================================================
#             UNCOMMENT FOR STATISTICAL+WAVELET/TIME+FFT DATA
# =============================================================================
#            statisticalWaveData = np.concatenate((totalData[:,0:12], totalData[:,18:27], totalData[:, 36:57]), axis=1)
#            TimeFFTData = np.concatenate((totalData[:,27:36], totalData[:, 12:18]), axis=1)
#            
#            x_train_stat_wave, x_test_stat_wave = train_test_split(statisticalWaveData, test_size=0.2)
#            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train_stat_wave.shape)
#            x_train_stat_wave_noisy = x_train_stat_wave + noise
#            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test_stat_wave.shape)
#            x_test_stat_wave_noisy = x_test_stat_wave + noise
#            
#            x_train_time_fft, x_test_time_fft = train_test_split(TimeFFTData, test_size=0.2)
#            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train_time_fft.shape)
#            x_train_time_fft_noisy = x_train_time_fft + noise
#            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test_time_fft.shape)
#            x_test_time_fft_noisy = x_test_time_fft + noise
#            
#            autoencoder_stat_wave, encoder_stat_wave = denoising_autoencoder(15, 25, 35, 42, x_train_stat_wave, x_train_stat_wave_noisy, x_test_stat_wave, x_test_stat_wave_noisy);
#            
#            autoencoder_time_fft, encoder_time_fft = denoising_autoencoder(5, 8, 12, 15, x_train_time_fft, x_train_time_fft_noisy, x_test_time_fft, x_test_time_fft_noisy);
#            
#            encoded_stats_wave = encoder_stat_wave.predict(statisticalWaveData)
#            encoded_time_fft = encoder_time_fft.predict(TimeFFTData)
#            
#            concat_encoded = np.concatenate((encoded_stats_wave, encoded_time_fft), axis=1)
#            
#            x_train_fused, x_test_fused = train_test_split(concat_encoded, test_size=0.2)
#            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train_fused.shape)
#            x_train_fused_noisy = x_train_fused + noise
#            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test_fused.shape)
#            x_test_fused_noisy = x_test_fused + noise
#            
#            autoencoder_fused, encoder_fused = denoising_autoencoder(7, 11, 16, 20, x_train_fused, x_train_fused_noisy, x_test_fused, x_test_fused_noisy);
#            
#            encoded_fused = encoder_fused.predict(concat_encoded)
#
#            np.savetxt("./resultsFusedDenoising3AEStatWavelet/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')
                       
# =============================================================================
#            UNCOMMENT FOR STATISTICAL+TIME/FFT+WAVELET DATA 
# =============================================================================
#        
#            statisticalTimeData = np.concatenate((totalData[:,0:12], totalData[:,18:36]), axis=1)
#            fftWaveletData = np.concatenate((totalData[:, 12:18], totalData[:, 36:57]), axis=1)
#            
#            x_train_stat_time, x_test_stat_time = train_test_split(statisticalTimeData, test_size=0.2)
#            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train_stat_time.shape)
#            x_train_stat_time_noisy = x_train_stat_time + noise
#            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test_stat_time.shape)
#            x_test_stat_time_noisy = x_test_stat_time + noise
#            
#            x_train_fft_wavelet, x_test_fft_wavelet = train_test_split(fftWaveletData, test_size=0.2)
#            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train_fft_wavelet.shape)
#            x_train_fft_wavelet_noisy = x_train_fft_wavelet + noise
#            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test_fft_wavelet.shape)
#            x_test_fft_wavelet_noisy = x_test_fft_wavelet + noise
#            
#            autoencoder_stat_time, encoder_stat_time = denoising_autoencoder(10, 16, 24, 30,x_train_stat_time, x_train_stat_time_noisy, x_test_stat_time, x_test_stat_time_noisy);
#            autoencoder_fft_wavelet, encoder_fft_wavelet = denoising_autoencoder(10, 15, 21, 27, x_train_fft_wavelet, x_train_fft_wavelet_noisy, x_test_fft_wavelet, x_test_fft_wavelet_noisy);
#            
#            encoded_stats_wave = encoder_stat_time.predict(statisticalTimeData)
#            encoded_fft_wavelet = encoder_fft_wavelet.predict(fftWaveletData)
#            
#            concat_encoded = np.concatenate((encoded_stats_wave, encoded_fft_wavelet), axis=1)
#            
#            x_train_fused, x_test_fused = train_test_split(concat_encoded, test_size=0.2)
#            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train_fused.shape)
#            x_train_fused_noisy = x_train_fused + noise
#            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test_fused.shape)
#            x_test_fused_noisy = x_test_fused + noise
#            
#            autoencoder_fused, encoder_fused = denoising_autoencoder(7, 12, 16, 20, x_train_fused, x_train_fused_noisy, x_test_fused, x_test_fused_noisy);
#            encoded_fused = encoder_fused.predict(concat_encoded)
#
#            np.savetxt("./resultsFusedDenoising3AEStatTime/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')


# =============================================================================
#            UNCOMMENT FOR STATISTICAL+FFT/TIME+WAVELET DATA 
# =============================================================================
        
            statisticalFFTData = totalData[:,0:27]
            TimeWaveletData = totalData[:,27:57]
            
            x_train_stat_fft, x_test_stat_fft = train_test_split(statisticalFFTData, test_size=0.2)
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train_stat_fft.shape)
            x_train_stat_fft_noisy = x_train_stat_fft + noise
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test_stat_fft.shape)
            x_test_stat_fft_noisy = x_test_stat_fft + noise
                       
            x_train_time_wavelet, x_test_time_wavelet = train_test_split(TimeWaveletData, test_size=0.2)
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train_time_wavelet.shape)
            x_train_time_wavelet_noisy = x_train_time_wavelet + noise
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test_time_wavelet.shape)
            x_test_time_wavelet_noisy = x_test_time_wavelet + noise
            
            autoencoder_stat_fft, encoder_stat_fft = denoising_autoencoder(10, 15, 21, 27, x_train_stat_fft, x_train_stat_fft_noisy, x_test_stat_fft, x_test_stat_fft_noisy);
            autoencoder_time_wavelet, encoder_time_wavelet = denoising_autoencoder(10, 16, 24, 30, x_train_time_wavelet, x_train_time_wavelet_noisy, x_test_time_wavelet, x_test_time_wavelet_noisy);
            
            encoded_stats_fft = encoder_stat_fft.predict(statisticalFFTData)
            encoded_time_wavelet = encoder_time_wavelet.predict(TimeWaveletData)
            
            concat_encoded = np.concatenate((encoded_stats_fft, encoded_time_wavelet), axis=1)
            
            x_train_fused, x_test_fused = train_test_split(concat_encoded, test_size=0.2)
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train_fused.shape)
            x_train_fused_noisy = x_train_fused + noise
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test_fused.shape)
            x_test_fused_noisy = x_test_fused + noise
            
            autoencoder_fused, encoder_fused = denoising_autoencoder(7, 12, 16, 20, x_train_fused, x_train_fused_noisy, x_test_fused, x_test_fused_noisy);
            encoded_fused = encoder_fused.predict(concat_encoded)

            np.savetxt("./resultsFusedDenoising3AEStatFFT/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')