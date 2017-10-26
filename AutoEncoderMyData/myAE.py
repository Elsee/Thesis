from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model
from keras import regularizers
import pandas as pd
import glob
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.regularizers import L1L2

# this is the size of our encoded representations
#encoding_dim = 25  # 76 floats -> compression of factor 2, assuming the input is 1140 floats
#lambda_l1 = 0.00001

# this is our input placeholder
#input_img = Input(shape=(57, 57))
#flat_img = Flatten()(input_img)

# "encoded" is the encoded representation of the input
#x = Dense(encoding_dim*3, activation='relu')(flat_img)
#x = Dense(encoding_dim*2, activation='relu')(x)
#encoded = Dense(encoding_dim, activation='linear', activity_regularizer=L1L2(lambda_l1))(x)
#encoded = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l2(0.01))(flat_img)

# create a placeholder for an encoded (32-dimensional) input
#input_encoded = Input(shape=(encoding_dim,))
#x = Dense(encoding_dim*2, activation='relu')(input_encoded)
#x = Dense(encoding_dim*3, activation='relu')(x)
#flat_decoded = Dense(3249, activation='sigmoid')(input_encoded)
#decoded = Reshape((57, 57))(flat_decoded)

# this model maps an input to its encoded representation
#encoder = Model(input_img, encoded)

# create the decoder model
#decoder = Model(input_encoded, decoded)

# this model maps an input to its reconstruction
#autoencoder = Model(input_img, decoder(encoder(input_img)))
#
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

def AE():
    encoding_dim = 25
    # Энкодер
    input_img = Input(shape=(57, 1)) # 28, 28, 1 - размерности строк, столбцов, фильтров одной картинки, без батч-размерности
    # Вспомогательный слой решейпинга
    flat_img = Flatten()(input_img)
    # Кодированное полносвязным слоем представление
    encoded = Dense(encoding_dim, activation='linear', activity_regularizer=L1L2(0,01))(flat_img)

    # Декодер
    # Раскодированное другим полносвязным слоем изображение
    input_encoded = Input(shape=(encoding_dim,))
    flat_decoded = Dense(57, activation='sigmoid')(input_encoded)
    decoded = Reshape((57, 1))(flat_decoded)
    
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)
    
    # create the decoder model
    decoder = Model(input_encoded, decoded)
    
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoder(encoder(input_img)))
    return encoder, decoder, autoencoder

#########################################################################################################
#####################################DATA LOADING########################################################
#########################################################################################################
#filenames = glob.glob('../myTrainingData/featuresOrig_*.csv')
#
#totalData = pd.DataFrame()
#    
#for item in filenames:
#    url = item
#    data = pd.read_csv(url, header = 0, engine='python')
#    totalData = pd.concat([totalData, data], ignore_index=True)

totalData = pd.read_csv('../myTrainingData/featuresOrig_Walking1.csv');
totalData.drop(["user"], axis=1, inplace=True)
totalData = np.asarray(totalData, dtype= np.float32);

#sc = StandardScaler()
#
#numRows = 57
#step = 20
#segments = []
#labels = []
#
#for i in range(0, len(totalData) - numRows, step):
#    label = stats.mode(totalData['user'][i: i + numRows])[0][0]
#    segments.append(np.array(sc.fit_transform(totalData.iloc[i: i + numRows,:-1])))
#    labels.append(label)
#    
#segments = np.asarray(segments, dtype= np.float32)
#labels = np.asarray(labels, dtype= np.float32)

##########################################################################################################
##########################################################################################################
##########################################################################################################


x_train, x_test = train_test_split(totalData, test_size=0.2)

x_train = np.reshape(x_train, (len(x_train), 57, 1))
x_test  = np.reshape(x_test,  (len(x_test),  57, 1))

print(x_train.shape)
print(x_test.shape)

autoencoder, encoder, decoder = AE()

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

autoencoder.summary()

test = autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


loss_history = test.history["loss"]
numpy_loss_history = np.array(loss_history)

print(test.history.keys())
# summarize history 
plt.plot(test.history['acc'], c='b', lw=1.5)
plt.plot(test.history['val_acc'], c='r', lw=1.5)
plt.plot(test.history['loss'], c='g', lw=1.5)
plt.plot(test.history['val_loss'], c='m', lw=1.5)

plt.title('model accuracy')
plt.ylabel('loss/accuracy')
plt.xlabel('epoch')
plt.legend(['train acc', 'test acc', 'train loss', 'test loss'], loc='upper left')
plt.tight_layout()
plt.savefig('./result.jpg', format='jpg')
plt.close()