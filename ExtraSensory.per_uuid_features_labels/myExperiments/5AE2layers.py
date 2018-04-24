from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import os

def ae(encoding_layer_dim, input_shape):
    # this is the size of our encoded representations
    encoding_dim = input_shape
    # this is our input placeholder
    input_img = Input(shape=(input_shape,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim*2, activation='relu')(input_img)
    encoded = Dense(encoding_dim, activation='linear',
                    activity_regularizer=regularizers.l2(0.00001))(encoded)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(encoding_dim*2, activation='relu')(encoded)
    decoded = Dense(input_shape, activation='sigmoid')(decoded)
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)
    # create a placeholder for an encoded input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-2]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    return autoencoder, encoder, decoder

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

allUsersData = os.listdir('./extractedFeatures')

features =  ["featuresFilt"]

for feature in features:
        
    for us in range(len(allUsersData)):
        usNum = us+1

        totalData = pd.read_csv('./extractedFeatures/' + feature + '_walking#' + str(usNum) + '.csv');
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
        
        #print(x_train.shape)
        #print(x_test.shape)
        
        autoencoder_stat, encoder_stat, decoder_stat = ae(10, 21);
        autoencoder_stat.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

        autoencoder_time, encoder_time, decoder_time = ae(4, 9);
        autoencoder_time.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

        autoencoder_fft, encoder_fft, decoder_fft = ae(3, 6);
        autoencoder_fft.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

        autoencoder_wavelet, encoder_wavelet, decoder_wavelet = ae(10, 21);
        autoencoder_wavelet.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

        test_stat = autoencoder_stat.fit(x_train_stat, x_train_stat,
                        epochs=3500,
                        batch_size=32,
                        shuffle=True,
                        validation_data=(x_test_stat, x_test_stat))

        test_time = autoencoder_time.fit(x_train_time, x_train_time,
                        epochs=3500,
                        batch_size=32,
                        shuffle=True,
                        validation_data=(x_test_time, x_test_time))
        
        test_fft = autoencoder_fft.fit(x_train_fft, x_train_fft,
                        epochs=3500,
                        batch_size=32,
                        shuffle=True,
                        validation_data=(x_test_fft, x_test_fft))
        
        test_wavelet = autoencoder_wavelet.fit(x_train_wavelet, x_train_wavelet,
                        epochs=3500,
                        batch_size=32,
                        shuffle=True,
                        validation_data=(x_test_wavelet, x_test_wavelet))
        
        # encode and decode some digits
        # note that we take them from the *test* set
        encoded_stats = encoder_stat.predict(statisticalData)
        encoded_time = encoder_time.predict(timeData)
        encoded_fft = encoder_fft.predict(fftData)
        encoded_wavelet = encoder_wavelet.predict(waveletData)
        
        concat_encoded = np.concatenate((encoded_stats, encoded_time, encoded_fft, encoded_wavelet), axis=1)
        
        x_train_fused, x_test_fused = train_test_split(concat_encoded, test_size=0.2)
        
        autoencoder_fused, encoder_fused, decoder_fused = ae(16, 57);
        autoencoder_fused.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
        
        test_fused = autoencoder_fused.fit(x_train_fused, x_train_fused,
                        epochs=3500,
                        batch_size=32,
                        shuffle=True,
                        validation_data=(x_test_fused, x_test_fused))
        
        encoded_fused = encoder_fused.predict(concat_encoded)
        np.savetxt("./results5AEsparseDeep/AEResult_" + feature + "_walking#" + str(usNum) +".csv", encoded_fused, delimiter=',')
        