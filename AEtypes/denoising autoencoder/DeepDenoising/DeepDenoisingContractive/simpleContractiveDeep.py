from keras.layers import Input, Dense
from keras.models import Model

import keras.backend as K

def contractive_autoencoder(encoding_layer_dim, input_shape, X, X_noisy, X_test, X_test_noisy, lam=1e-3):
    N = input_shape
    N_hidden = encoding_layer_dim

    inputs = Input(shape=(N,))
    
    encoded = Dense(N_hidden*2, activation='sigmoid')(inputs)
    encoded = Dense(N_hidden, activation='sigmoid', name='encoded')(encoded)
    
    outputs = Dense(N_hidden*2, activation='sigmoid')(encoded)
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
    model.fit(X_noisy, X, 
              batch_size=32, 
              epochs=400,
              shuffle=True,
              validation_data=(X_test_noisy, X_test))

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
            
            x_train, x_test = train_test_split(totalData, test_size=0.2)
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
            x_train_noisy = x_train + noise
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
            x_test_noisy = x_test + noise

            autoencoder, encoder = contractive_autoencoder(28, 57, x_train, x_train_noisy, x_test, x_test_noisy)
            
            encoded = encoder.predict(totalData)
            
            np.savetxt("./resultsSimpleContractive/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded, delimiter=',')
