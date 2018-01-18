import numpy as np

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics

def variational_autoencoder(encoding_layer_dim, input_shape, X, X_test):
    x = Input(shape=(input_shape,))
#    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(encoding_layer_dim, activation='relu')(x)
    z_log_var = Dense(encoding_layer_dim, activation='relu')(x)
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], encoding_layer_dim), mean=0.,
                                  stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(encoding_layer_dim,))([z_mean, z_log_var])
    
    # we instantiate these layers separately so as to reuse them later
#    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(input_shape, activation='sigmoid')
#    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(z)
    
    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)
    
        def vae_loss(self, x, x_decoded_mean):
            xent_loss = input_shape * metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)
    
        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            # We won't actually use the output.
            return x
    
    y = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, y)
    vae.compile(optimizer='rmsprop', loss=None, metrics=['accuracy'])
    
    vae.fit(X, X, 
              batch_size=32, 
              epochs=400,
              shuffle=True,
              validation_data=(X_test, None))
    
    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)

    return vae, encoder

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
            statisticalWaveData = np.concatenate((totalData[:,0:12], totalData[:,18:27], totalData[:, 36:57]), axis=1)
            TimeFFTData = np.concatenate((totalData[:,27:36], totalData[:, 12:18]), axis=1)
            
            x_train_stat_wave, x_test_stat_wave = train_test_split(statisticalWaveData, test_size=0.2)
            x_train_time_fft, x_test_time_fft = train_test_split(TimeFFTData, test_size=0.2)
            
            autoencoder_stat_wave, encoder_stat_wave = variational_autoencoder(21, 42, x_train_stat_wave, x_test_stat_wave);
            
            autoencoder_time_fft, encoder_time_fft = variational_autoencoder(8, 15, x_train_time_fft, x_test_time_fft);
            
            encoded_stats_wave = encoder_stat_wave.predict(statisticalWaveData)
            encoded_time_fft = encoder_time_fft.predict(TimeFFTData)
            
            concat_encoded = np.concatenate((encoded_stats_wave, encoded_time_fft), axis=1)
            
            x_train_fused, x_test_fused = train_test_split(concat_encoded, test_size=0.2)
            
            autoencoder_fused, encoder_fused = variational_autoencoder(15, 29, x_train_fused, x_test_fused);
            
            encoded_fused = encoder_fused.predict(concat_encoded)

            np.savetxt("./resultsFusedVariational3AEStatWavelet/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')
                       
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
#            autoencoder_stat_time, encoder_stat_time = variational_autoencoder(16, 30,x_train_stat_time, x_test_stat_time);
#            autoencoder_fft_wavelet, encoder_fft_wavelet = variational_autoencoder(16, 27, x_train_fft_wavelet, x_test_fft_wavelet);
#            
#            encoded_stats_wave = encoder_stat_time.predict(statisticalTimeData)
#            encoded_fft_wavelet = encoder_fft_wavelet.predict(fftWaveletData)
#            
#            concat_encoded = np.concatenate((encoded_stats_wave, encoded_fft_wavelet), axis=1)
#            
#            x_train_fused, x_test_fused = train_test_split(concat_encoded, test_size=0.2)
#            
#            autoencoder_fused, encoder_fused = variational_autoencoder(16, 32, x_train_fused, x_test_fused);
#            encoded_fused = encoder_fused.predict(concat_encoded)
#
#            np.savetxt("./resultsFusedVariational3AEStatTime/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')


# =============================================================================
#            UNCOMMENT FOR STATISTICAL+FFT/TIME+WAVELET DATA 
# =============================================================================
        
#            statisticalFFTData = totalData[:,0:27]
#            TimeWaveletData = totalData[:,27:57]
#            
#            x_train_stat_fft, x_test_stat_fft = train_test_split(statisticalFFTData, test_size=0.2)
#            x_train_time_wavelet, x_test_time_wavelet = train_test_split(TimeWaveletData, test_size=0.2)
#            
#            autoencoder_stat_fft, encoder_stat_fft = variational_autoencoder(14, 27, x_train_stat_fft, x_test_stat_fft);
#            autoencoder_time_wavelet, encoder_time_wavelet = variational_autoencoder(15, 30, x_train_time_wavelet, x_test_time_wavelet);
#            
#            encoded_stats_fft = encoder_stat_fft.predict(statisticalFFTData)
#            encoded_time_wavelet = encoder_time_wavelet.predict(TimeWaveletData)
#            
#            concat_encoded = np.concatenate((encoded_stats_fft, encoded_time_wavelet), axis=1)
#            
#            x_train_fused, x_test_fused = train_test_split(concat_encoded, test_size=0.2)
#            
#            autoencoder_fused, encoder_fused = variational_autoencoder(15, 29, x_train_fused, x_test_fused);
#            encoded_fused = encoder_fused.predict(concat_encoded)
#
#            np.savetxt("./resultsFusedVariational3AEStatFFT/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')