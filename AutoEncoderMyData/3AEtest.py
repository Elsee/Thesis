from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

def ae(encoding_layer_dim, input_shape):
    # this is the size of our encoded representations
    encoding_dim = encoding_layer_dim
    # this is our input placeholder
    input_img = Input(shape=(input_shape,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='linear',
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

            totalData = pd.read_csv('../myTrainingData/' + feature + '_' + act + '#' + str(us) + '.csv');
            totalData.drop(["user"], axis=1, inplace=True)
            totalData = sc.fit_transform(np.asarray(totalData, dtype= np.float32));
            
            statisticalWaveData = np.concatenate((totalData[:,0:12], totalData[:, 36:57]), axis=1)
            TimeFFTData = np.concatenate((totalData[:,18:36], totalData[:, 12:18]), axis=1)
            
            x_train_stat_wave, x_test_stat_wave = train_test_split(statisticalWaveData, test_size=0.2)
            x_train_time_fft, x_test_time_fft = train_test_split(TimeFFTData, test_size=0.2)
            
            x_train_stat_wave = x_train_stat_wave.reshape((len(x_train_stat_wave), -1))
            x_test_stat_wave = x_test_stat_wave.reshape((len(x_test_stat_wave), -1))
            
            x_train_time_fft = x_train_time_fft.reshape((len(x_train_time_fft), -1))
            x_test_time_fft = x_test_time_fft.reshape((len(x_test_time_fft), -1))
            
            #print(x_train.shape)
            #print(x_test.shape)
            
            autoencoder_stat_wave, encoder_stat_wave, decoder_stat_wave = ae(17, 33);
            autoencoder_stat_wave.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
            
            autoencoder_time_fft, encoder_time_fft, decoder_time_fft = ae(12, 24);
            autoencoder_time_fft.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
            
            test_stat_time = autoencoder_stat_wave.fit(x_train_stat_wave, x_train_stat_wave,
                            epochs=3500,
                            batch_size=32,
                            shuffle=True,
                            validation_data=(x_test_stat_wave, x_test_stat_wave))
            
            test_fft_wavelet = autoencoder_time_fft.fit(x_train_time_fft, x_train_time_fft,
                            epochs=3500,
                            batch_size=32,
                            shuffle=True,
                            validation_data=(x_test_time_fft, x_test_time_fft))
            
            # encode and decode some digits
            # note that we take them from the *test* set
            encoded_stats_wave = encoder_stat_wave.predict(statisticalWaveData)
            encoded_time_fft = encoder_time_fft.predict(TimeFFTData)
            
            concat_encoded = np.concatenate((encoded_stats_wave, encoded_time_fft), axis=1)
            
            x_train_fused, x_test_fused = train_test_split(concat_encoded, test_size=0.2)
            
            autoencoder_fused, encoder_fused, decoder_fused = ae(15, 29);
            autoencoder_fused.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
            
            test_fused = autoencoder_fused.fit(x_train_fused, x_train_fused,
                            epochs=3500,
                            batch_size=32,
                            shuffle=True,
                            validation_data=(x_test_fused, x_test_fused))
            
            encoded_fused = encoder_fused.predict(concat_encoded)
            np.savetxt("./results3AEStatWave/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')
            
            
#            if (feature == "featuresOrig" and act == "Jogging"):
#                import matplotlib.pyplot as plt
#                
#                def result_graph(test_type, name):
#                    # summarize history 
#                    plt.plot(test_type.history['acc'], c='b', lw=1.5)
#                    plt.plot(test_type.history['val_acc'], c='r', lw=1.5)
#                    plt.plot(test_type.history['loss'], c='g', lw=1.5)
#                    plt.plot(test_type.history['val_loss'], c='m', lw=1.5)
#                    
#                    plt.title(name + str(us) + ' result')
#                    plt.ylabel('loss/accuracy')
#                    plt.xlabel('epoch')
#                    plt.legend(['train acc', 'test acc', 'train loss', 'test loss'], loc='upper left')
#                    plt.tight_layout()
#                    plt.savefig('./results3AE/graphs/' + name + str(us) + '_result.jpg', format='jpg')
#                    plt.close()
#                    
#                result_graph(test_stat_time, ("statistical and time"))
#                result_graph(test_fft_wavelet, "fft and wavelet")
#                result_graph(test_fused, "total")