from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

def ae(encoding_layer_dim, input_shape):
    # this is the size of our encoded representations
    encoding_dim = encoding_layer_dim
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

users = [1,2,3,4,5,6]
activities = ["Jogging", "Running", "Walking down-stairs", "Walking up-stairs", "Walking"]
features =  ["featuresOrig", "featuresFilt"]

for feature in features:

    for act in activities:
        
        for us in users:

            totalData = pd.read_csv('../myTrainingData/' + feature + '_' + act + '#' + str(us) + '.csv');
            totalData.drop(["user"], axis=1, inplace=True)
            totalData = sc.fit_transform(np.asarray(totalData, dtype= np.float32));
            
            x_train, x_test = train_test_split(totalData, test_size=0.2)
            
            x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
            x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
            
            autoencoder, encoder, decoder = ae(16, 57);
            autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
            
            test_fused = autoencoder.fit(x_train, x_train,
                            epochs=3500,
                            batch_size=256,
                            shuffle=True,
                            validation_data=(x_test, x_test))
            
            encoded_fused = encoder.predict(totalData)
            np.savetxt("./results1AEdeep/AEResult_" + feature + "_" + act + '#' + str(us) +".csv", encoded_fused, delimiter=',')

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