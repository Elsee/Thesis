import numpy as np

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics

def variational_autoencoder(encoding_layer_dim, intermediate_dim1, intermediate_dim2, input_shape, X, X_test):
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
            totalData = pd.read_csv('../../../myTrainingData/' + feature + '_' + act + '#' + str(us) + '.csv');
            totalData.drop(["user"], axis=1, inplace=True)
            totalData = sc.fit_transform(np.asarray(totalData, dtype= np.float32));
            
            x_train, x_test = train_test_split(totalData, test_size=0.2)

            autoencoder, encoder = variational_autoencoder(18, 28, 40, 57, x_train, x_test)
            
            encoded = encoder.predict(totalData)
            
            np.savetxt("./resultsSimpleVariational/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded, delimiter=',')
