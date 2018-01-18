import numpy as np

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics

def variational_autoencoder(encoding_layer_dim, intermediate_dim1, intermediate_dim2, input_shape, X, X_noisy, X_test, X_test_noisy):
    x = Input(shape=(input_shape,))
    h1 = Dense(intermediate_dim2, activation='relu')(x)
    h2 = Dense(intermediate_dim1, activation='relu')(h1)
    z_mean = Dense(encoding_layer_dim, activation='relu')(h2)
    z_log_var = Dense(encoding_layer_dim, activation='relu')(h2)
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], encoding_layer_dim), mean=0.,
                                  stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(encoding_layer_dim,))([z_mean, z_log_var])
    
    # we instantiate these layers separately so as to reuse them later
    decoder_h1 = Dense(intermediate_dim1, activation='relu')
    decoder_h2 = Dense(intermediate_dim2, activation='relu')
    decoder_mean = Dense(input_shape, activation='sigmoid')
    h_decoded1 = decoder_h1(z)
    h_decoded2 = decoder_h2(h_decoded1)    
    x_decoded_mean = decoder_mean(h_decoded2)
    
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
    
    vae.fit(X_noisy, X, 
              batch_size=32, 
              epochs=400,
              shuffle=True,
              validation_data=(X_test_noisy, X_test))
    
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
            totalData = pd.read_csv('../../../../myTrainingData/' + feature + '_' + act + '#' + str(us) + '.csv');
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
            
            autoencoder_stat, encoder_stat = variational_autoencoder(7, 12, 17, 21, x_train_stat, x_train_stat_noisy, x_test_stat, x_test_stat_noisy);

            autoencoder_time, encoder_time = variational_autoencoder(3, 5, 7, 9, x_train_time, x_train_time_noisy, x_test_time, x_test_time_noisy);

            autoencoder_fft, encoder_fft = variational_autoencoder(2, 4, 5, 6, x_train_fft, x_train_fft_noisy, x_test_fft, x_test_fft_noisy);

            autoencoder_wavelet, encoder_wavelet = variational_autoencoder(8, 12, 16, 21, x_train_wavelet, x_train_wavelet_noisy, x_test_wavelet, x_test_wavelet_noisy);
            
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

            autoencoder_fused, encoder_fused = variational_autoencoder(8, 12, 16, 20, x_train_fused, x_train_fused_noisy, x_test_fused, x_test_fused_noisy);

            encoded_fused = encoder_fused.predict(concat_encoded)
            np.savetxt("./resultsFusedVariational5AE/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')