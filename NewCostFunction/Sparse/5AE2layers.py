from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.metrics.pairwise import euclidean_distances
from keras import backend as K
from keras import metrics

def ae(encoding_layer_dim, input_shape, X, X_test):
    encoding_dim = input_shape
    classInstance = Input(shape=(input_shape,))
    encoded = Dense(encoding_dim*2, activation='relu')(classInstance)
    encoded = Dense(encoding_dim, activation='linear',
                    activity_regularizer=regularizers.l2(0.00001))(encoded)
    decoded = Dense(encoding_dim*2, activation='relu')(encoded)
    decoded = Dense(input_shape, activation='sigmoid')(decoded)
    autoencoder = Model(classInstance, decoded)
    encoder = Model(classInstance, encoded)
    
    def custom_loss(classInstance, decoded):
        mse_loss = K.mean(K.square(decoded - classInstance), axis=-1)
        intra_spread_loss = K.mean(K.sqrt(K.sum((K.mean(classInstance, axis=0) - classInstance)**2, axis=1)))
        return K.mean(mse_loss + intra_spread_loss)
    
    autoencoder.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])
    
    autoencoder.fit(X, X, 
              batch_size=input_shape, 
              epochs=200,
              shuffle=True,
              validation_data=(X_test, X_test))

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
            totalData = pd.read_csv('../../myTrainingData/' + feature + '_' + act + '#' + str(us) + '.csv');
            totalData.drop(["user"], axis=1, inplace=True)
            totalData = sc.fit_transform(np.asarray(totalData, dtype= np.float32));
            
            statisticalData = np.concatenate((totalData[:,0:12], totalData[:,18:27]), axis=1)
            timeData = totalData[:,27:36]
            fftData = totalData[:, 12:18]
            waveletData = totalData[:, 36:57]
            
            x_train_stat, x_test_stat = train_test_split(statisticalData, test_size=0.2)
            x_train_time, x_test_time = train_test_split(timeData, test_size=0.2)
            x_train_fft, x_test_fft = train_test_split(fftData, test_size=0.2)
            x_train_wavelet, x_test_wavelet = train_test_split(waveletData, test_size=0.2)

            x_train_stat = x_train_stat.reshape((len(x_train_stat), np.prod(x_train_stat.shape[1:])))
            x_test_stat = x_test_stat.reshape((len(x_test_stat), np.prod(x_test_stat.shape[1:])))

            x_train_time = x_train_time.reshape((len(x_train_time), np.prod(x_train_time.shape[1:])))
            x_test_time = x_test_time.reshape((len(x_test_time), np.prod(x_test_time.shape[1:])))

            x_train_fft = x_train_fft.reshape((len(x_train_fft), np.prod(x_train_fft.shape[1:])))
            x_test_fft = x_test_fft.reshape((len(x_test_fft), np.prod(x_test_fft.shape[1:])))

            x_train_wavelet = x_train_wavelet.reshape((len(x_train_wavelet), np.prod(x_train_wavelet.shape[1:])))
            x_test_wavelet = x_test_wavelet.reshape((len(x_test_wavelet), np.prod(x_test_wavelet.shape[1:])))
            
            autoencoder_stat, encoder_stat = ae(10, 21, x_train_stat, x_test_stat);

            autoencoder_time, encoder_time = ae(4, 9, x_train_time, x_test_time);
        
            autoencoder_fft, encoder_fft = ae(3, 6, x_train_fft, x_test_fft);

            autoencoder_wavelet, encoder_wavelet = ae(10, 21, x_train_wavelet, x_test_wavelet);

            encoded_stats = encoder_stat.predict(statisticalData)
            encoded_time = encoder_time.predict(timeData)
            encoded_fft = encoder_fft.predict(fftData)
            encoded_wavelet = encoder_wavelet.predict(waveletData)
            
            concat_encoded = np.concatenate((encoded_stats, encoded_time, encoded_fft, encoded_wavelet), axis=1)
            
            x_train_fused, x_test_fused = train_test_split(concat_encoded, test_size=0.2)
            
            autoencoder_fused, encoder_fused = ae(16, 57, x_train_fused, x_test_fused);
            
            encoded_fused = encoder_fused.predict(concat_encoded)
            np.savetxt("./results5AEdeep/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')
            