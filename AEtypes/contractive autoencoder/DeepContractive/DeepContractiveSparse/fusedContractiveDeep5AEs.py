from keras.layers import Input, Dense
from keras.models import Model

import keras.backend as K
from keras import regularizers

def contractive_autoencoder(hidden1, hidden2, input_shape, X, X_test, lam=1e-3):
    N = input_shape
    N_hidden = hidden1

    inputs = Input(shape=(N,))
    
    
    encoded = Dense(hidden2, activation='relu')(inputs)
#    encoded = Dense(hidden2, activation='relu')(encoded)
    encoded = Dense(N_hidden, activation='linear',
                    activity_regularizer=regularizers.l2(0.00001), name='encoded')(encoded)
    
    outputs = Dense(hidden2, activation='relu')(encoded)
#    outputs = Dense(hidden3, activation='relu')(outputs)
    outputs = Dense(input_shape, activation='sigmoid')(outputs)


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
            totalData = pd.read_csv('../../../../myTrainingData/' + feature + '_' + act + '#' + str(us) + '.csv');
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
            
            autoencoder_stat, encoder_stat = contractive_autoencoder(12, 17, 21, x_train_stat, x_test_stat);

            autoencoder_time, encoder_time = contractive_autoencoder(5, 7, 9, x_train_time, x_test_time);

            autoencoder_fft, encoder_fft = contractive_autoencoder(4, 5, 6, x_train_fft, x_test_fft);

            autoencoder_wavelet, encoder_wavelet = contractive_autoencoder(12, 16, 21, x_train_wavelet, x_test_wavelet);
            
            encoded_stats = encoder_stat.predict(statisticalData)
            encoded_time = encoder_time.predict(timeData)
            encoded_fft = encoder_fft.predict(fftData)
            encoded_wavelet = encoder_wavelet.predict(waveletData)

            concat_encoded = np.concatenate((encoded_stats, encoded_time, encoded_fft, encoded_wavelet), axis=1)

            x_train_fused, x_test_fused = train_test_split(concat_encoded, test_size=0.2)

            autoencoder_fused, encoder_fused = contractive_autoencoder(18, 26, 33, x_train_fused, x_test_fused);

            encoded_fused = encoder_fused.predict(concat_encoded)
            np.savetxt("./resultsFusedContractive5AE/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')