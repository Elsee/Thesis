from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras import backend as K

def ae(encoding_layer_dim, input_shape, X, X_interpolated, X_test, X_test_interpolated):
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
    
    autoencoder.compile(loss=custom_loss, optimizer='adadelta', metrics=['accuracy'])
    
    autoencoder.fit(X_interpolated, X_interpolated, 
              batch_size=input_shape, 
              epochs=100,
              shuffle=True,
              validation_data=(X_test_interpolated, X_test_interpolated))
    
    return autoencoder, encoder

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

sc = StandardScaler()

users = [1,2,3,4,5,6]
activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]
features =  ["featuresFilt"]

NEIGHBOURS_AMOUNT = 10 #the amount of needed neighbours
LAMBDA = 0.5

def KNN_Interpolation(featureSpace):
    space_width = featureSpace.shape[1]
    final_space = pd.DataFrame(columns=range(space_width))
    nbrs = NearestNeighbors(n_neighbors=NEIGHBOURS_AMOUNT).fit(featureSpace)
    distances, indices = nbrs.kneighbors(featureSpace)
    
    for idx, val in enumerate(featureSpace):
        for i in range(NEIGHBOURS_AMOUNT):
            needed_vector = indices[idx][i]
            interpolated_vector = (featureSpace[needed_vector]-val)*0.5 + val
            interpolated_vector = interpolated_vector.reshape((1, space_width))
            final_space = np.r_[final_space, interpolated_vector]
    
    final_space = final_space.astype(float)
    return final_space

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
        
            x_train_stat_interpolated = KNN_Interpolation(x_train_stat)
            x_test_stat_interpolated = KNN_Interpolation(x_test_stat)
            
            x_train_time, x_test_time = train_test_split(timeData, test_size=0.2)
            
            x_train_time_interpolated = KNN_Interpolation(x_train_time)
            x_test_time_interpolated = KNN_Interpolation(x_test_time)
            
            x_train_fft, x_test_fft = train_test_split(fftData, test_size=0.2)
            
            x_train_fft_interpolated = KNN_Interpolation(x_train_fft)
            x_test_fft_interpolated = KNN_Interpolation(x_test_fft)
            
            x_train_wavelet, x_test_wavelet = train_test_split(waveletData, test_size=0.2)
            
            x_train_wavelet_interpolated = KNN_Interpolation(x_train_wavelet)
            x_test_wavelet_interpolated = KNN_Interpolation(x_test_wavelet)
            
            autoencoder_stat, encoder_stat = ae(10, 21, x_train_stat, x_train_stat_interpolated, x_test_stat, x_test_stat_interpolated);

            autoencoder_time, encoder_time = ae(4, 9, x_train_time, x_train_time_interpolated, x_test_time, x_test_time_interpolated);

            autoencoder_fft, encoder_fft = ae(3, 6, x_train_fft, x_train_fft_interpolated, x_test_fft, x_test_fft_interpolated);

            autoencoder_wavelet, encoder_wavelet = ae(10, 21, x_train_wavelet, x_train_wavelet_interpolated, x_test_wavelet, x_test_wavelet_interpolated);
            
            encoded_stats = encoder_stat.predict(statisticalData)
            encoded_time = encoder_time.predict(timeData)
            encoded_fft = encoder_fft.predict(fftData)
            encoded_wavelet = encoder_wavelet.predict(waveletData)

            concat_encoded = np.concatenate((encoded_stats, encoded_time, encoded_fft, encoded_wavelet), axis=1)

            x_train_fused, x_test_fused = train_test_split(concat_encoded, test_size=0.2)
            
            x_train_fused_interpolated = KNN_Interpolation(x_train_fused)
            x_test_fused_interpolated = KNN_Interpolation(x_test_fused)

            autoencoder_fused, encoder_fused = ae(16, 57, x_train_fused, x_train_fused_interpolated, x_test_fused, x_test_fused_interpolated);

            encoded_fused = encoder_fused.predict(concat_encoded)
            np.savetxt("./resultsFusedInterpolated5AE/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')