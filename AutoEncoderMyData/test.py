from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

def ae(encoding_layer_dim, input_shape):
    # this is the size of our encoded representations
    encoding_dim = encoding_layer_dim
    # this is our input placeholder
    input_img = Input(shape=(input_shape,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu',
                    activity_regularizer=regularizers.l2(0.00001))(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_shape, activation='sigmoid')(encoded)
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)
    # create a placeholder for an encoded input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    return autoencoder, encoder, decoder

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

            totalData = pd.read_csv('../myTrainingData/' + feature + '_' + act + str(us) + '.csv');
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
                            epochs=6000,
                            batch_size=256,
                            shuffle=True,
                            validation_data=(x_test_stat, x_test_stat))
            
            test_time = autoencoder_time.fit(x_train_time, x_train_time,
                            epochs=6000,
                            batch_size=256,
                            shuffle=True,
                            validation_data=(x_test_time, x_test_time))
            
            test_fft = autoencoder_fft.fit(x_train_fft, x_train_fft,
                            epochs=6000,
                            batch_size=256,
                            shuffle=True,
                            validation_data=(x_test_fft, x_test_fft))
            
            test_wavelet = autoencoder_wavelet.fit(x_train_wavelet, x_train_wavelet,
                            epochs=6000,
                            batch_size=256,
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
            
            autoencoder_fused, encoder_fused, decoder_fused = ae(15, 27);
            autoencoder_fused.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
            
            test_fused = autoencoder_fused.fit(x_train_fused, x_train_fused,
                            epochs=6000,
                            batch_size=256,
                            shuffle=True,
                            validation_data=(x_test_fused, x_test_fused))
            
            encoded_fused = encoder_fused.predict(concat_encoded)
            np.savetxt("./AEResult_" + feature + "_" + act + str(us) +".csv", encoded_fused, delimiter=',')
            
            plt.plot(waveletData, c='r', lw=1.5)
            plt.title('Wawelet features')
            
            #import matplotlib.pyplot as plt
            #
            #def result_graph(test_type, name):
            #    # summarize history 
            #    plt.plot(test_type.history['acc'], c='b', lw=1.5)
            #    plt.plot(test_type.history['val_acc'], c='r', lw=1.5)
            #    plt.plot(test_type.history['loss'], c='g', lw=1.5)
            #    plt.plot(test_type.history['val_loss'], c='m', lw=1.5)
            #    
            #    plt.title('model accuracy')
            #    plt.ylabel('loss/accuracy')
            #    plt.xlabel('epoch')
            #    plt.legend(['train acc', 'test acc', 'train loss', 'test loss'], loc='upper left')
            #    plt.tight_layout()
            #    plt.savefig('./' + name + '_result.jpg', format='jpg')
            #    plt.close()
            #    
            #result_graph(test_stat, "statistical")
            #result_graph(test_time, "time")
            #result_graph(test_fft, "fft")
            #result_graph(test_wavelet, "wavelet")
            #result_graph(test_fused, "total")