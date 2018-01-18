from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

def denoising_autoencoder(encoding_layer_dim, intermediate_dim1, intermediate_dim2, input_shape, X, X_noisy, X_test, X_test_noisy):
    # this is the size of our encoded representations
    encoding_dim = encoding_layer_dim
    # this is our input placeholder
    input_img = Input(shape=(input_shape,))
    # "encoded" is the encoded representation of the input
    h1 = Dense(intermediate_dim2, activation='relu')(input_img)
    h2 = Dense(intermediate_dim1, activation='relu')(h1)
    encoded = Dense(encoding_dim, activation='relu', name='encoded')(h2)
    # "decoded" is the lossy reconstruction of the input
    decoded_h2 = Dense(intermediate_dim1, activation='sigmoid')(encoded)
    decoded_h1 = Dense(intermediate_dim2, activation='sigmoid')(decoded_h2)
    decoded = Dense(input_shape, activation='sigmoid')(decoded_h1)
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)
    
    autoencoder.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    
    autoencoder.fit(X_noisy, X, 
              batch_size=32, 
              epochs=400,
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
features =  ["featuresOrig", "featuresFilt"]

for feature in features:

    for act in activities:
        
        for us in users:
            totalData = pd.read_csv('../../../myTrainingData/' + feature + '_' + act + '#' + str(us) + '.csv');
            totalData.drop(["user"], axis=1, inplace=True)
            totalData = sc.fit_transform(np.asarray(totalData, dtype= np.float32));
            
            x_train, x_test = train_test_split(totalData, test_size=0.2)

            noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
            x_train_noisy = x_train + noise
            noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
            x_test_noisy = x_test + noise
            
            autoencoder, encoder = denoising_autoencoder(18, 28, 40, 57, x_train, x_train_noisy, x_test, x_test_noisy)
            
            encoded = encoder.predict(totalData)
            
            np.savetxt("./resultsSimpleDenoising/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded, delimiter=',')
