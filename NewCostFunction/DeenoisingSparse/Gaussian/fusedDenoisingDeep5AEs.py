from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras import backend as K

def denoising_autoencoder(encoding_layer_dim, input_shape, X, X_noisy, X_test, X_test_noisy):
    # this is the size of our encoded representations
    encoding_dim = input_shape
    # this is our input placeholder
    input_img = Input(shape=(input_shape,))
    # "encoded" is the encoded representation of the input
    h1 = Dense(encoding_dim*2, activation='relu')(input_img)
    encoded = Dense(encoding_dim, activation='linear',
                    activity_regularizer=regularizers.l2(0.00001), name='encoded')(h1)
    # "decoded" is the lossy reconstruction of the input
    decoded_h2 = Dense(encoding_dim*2, activation='relu')(encoded)
    decoded = Dense(input_shape, activation='sigmoid')(decoded_h2)
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)
    
    def custom_loss(classInstance, decoded):
        mse_loss = K.mean(K.square(decoded - classInstance), axis=-1)
        W = K.variable(value=autoencoder.get_layer('encoded').get_weights()[0])
        intra_spread_loss = K.mean(K.sqrt((K.square(K.mean(W, axis=0) - W)).sum(1)), axis=-1)
        return K.mean(mse_loss + intra_spread_loss)
    
    autoencoder.compile(optimizer='adadelta', loss=custom_loss, metrics=['accuracy'])
    
    autoencoder.fit(X_noisy, X, 
              batch_size=input_shape, 
              epochs=100,
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
features =  ["featuresFilt"]

for feature in features:

    for act in activities:
        
        for us in users:
            totalData = pd.read_csv('../../../myTrainingData/' + feature + '_' + act + '#' + str(us) + '.csv');
            totalData.drop(["user"], axis=1, inplace=True)
            totalData = sc.fit_transform(np.asarray(totalData, dtype= np.float32));
            
            statisticalData = np.concatenate((totalData[:,0:12], totalData[:,18:27]), axis=1)
            timeData = totalData[:,27:36]
            fftData = totalData[:, 12:18]
            waveletData = totalData[:, 36:57]

            x_train_stat, x_test_stat = train_test_split(statisticalData, test_size=0.2)
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train_stat.shape)
            x_train_stat_noisy = x_train_stat + noise
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test_stat.shape)
            x_test_stat_noisy = x_test_stat + noise
            
            x_train_time, x_test_time = train_test_split(timeData, test_size=0.2)
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train_time.shape)
            x_train_time_noisy = x_train_time + noise
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test_time.shape)
            x_test_time_noisy = x_test_time + noise
            
            x_train_fft, x_test_fft = train_test_split(fftData, test_size=0.2)
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train_fft.shape)
            x_train_fft_noisy = x_train_fft + noise
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test_fft.shape)
            x_test_fft_noisy = x_test_fft + noise
            
            x_train_wavelet, x_test_wavelet = train_test_split(waveletData, test_size=0.2)
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train_wavelet.shape)
            x_train_wavelet_noisy = x_train_wavelet + noise
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test_wavelet.shape)
            x_test_wavelet_noisy = x_test_wavelet + noise
            
            autoencoder_stat, encoder_stat = denoising_autoencoder(10, 21, x_train_stat, x_train_stat_noisy, x_test_stat, x_test_stat_noisy);

            autoencoder_time, encoder_time = denoising_autoencoder(4, 9, x_train_time, x_train_time_noisy, x_test_time, x_test_time_noisy);

            autoencoder_fft, encoder_fft = denoising_autoencoder(3, 6, x_train_fft, x_train_fft_noisy, x_test_fft, x_test_fft_noisy);

            autoencoder_wavelet, encoder_wavelet = denoising_autoencoder(10, 21, x_train_wavelet, x_train_wavelet_noisy, x_test_wavelet, x_test_wavelet_noisy);
            
            encoded_stats = encoder_stat.predict(statisticalData)
            encoded_time = encoder_time.predict(timeData)
            encoded_fft = encoder_fft.predict(fftData)
            encoded_wavelet = encoder_wavelet.predict(waveletData)

            concat_encoded = np.concatenate((encoded_stats, encoded_time, encoded_fft, encoded_wavelet), axis=1)

            x_train_fused, x_test_fused = train_test_split(concat_encoded, test_size=0.2)
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train_fused.shape)
            x_train_fused_noisy = x_train_fused + noise
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test_fused.shape)
            x_test_fused_noisy = x_test_fused + noise

            autoencoder_fused, encoder_fused = denoising_autoencoder(16, 57, x_train_fused, x_train_fused_noisy, x_test_fused, x_test_fused_noisy);

            encoded_fused = encoder_fused.predict(concat_encoded)
            np.savetxt("./resultsFusedDenoising5AE/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')