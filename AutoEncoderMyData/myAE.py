from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model
from keras.regularizers import L1L2
import pandas as pd
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# this is the size of our encoded representations
encoding_dim = 960  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
lambda_l1 = 0.00001

# this is our input placeholder
input_img = Input(shape=(3923, 57, 1))
flat_img = Flatten()(input_img)
# "encoded" is the encoded representation of the input
x = Dense(encoding_dim*3, activation='relu')(flat_img)
x = Dense(encoding_dim*2, activation='relu')(x)
encoded = Dense(encoding_dim, activation='linear', activity_regularizer=L1L2(lambda_l1))(x)

# create a placeholder for an encoded (32-dimensional) input
input_encoded = Input(shape=(encoding_dim,))
x = Dense(encoding_dim*2, activation='relu')(input_encoded)
x = Dense(encoding_dim*3, activation='relu')(x)
flat_decoded = Dense(223611, activation='sigmoid')(x)
decoded = Reshape((3923, 57, 1))(flat_decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create the decoder model
decoder = Model(input_encoded, decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoder(encoder(input_img)))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

filenames = glob.glob('../myTrainingData/featuresOrig_*.csv')

totalData = pd.DataFrame()
    
for item in filenames:
    # Load current dataset
    url = item
    #choose only accelerometer data
    data = pd.read_csv(url, header = 0, engine='python')
    totalData = pd.concat([totalData, data], ignore_index=True)

totalData = totalData.sort_values(['user'], ascending = 1)

totalData.set_index(keys=['user'], drop=False,inplace=True)
labels=totalData['user'].unique().tolist()

usersData = {}
usersDataLen = []

for i in labels:
     usersData["user{0}".format(i)] = totalData.loc[totalData.user==i]

     usersDataLen.append(usersData["user{0}".format(i)].shape[0])
     
minUserSamples = min(usersDataLen)
segments = []
sc = StandardScaler()

for i in labels:
    usersData["user{0}".format(i)] = usersData["user{0}".format(i)].head(n=minUserSamples)
    usersData["user{0}".format(i)] = sc.fit_transform(usersData["user{0}".format(i)])

    segments.append(np.array(usersData["user{0}".format(i)][:,:-1]))

labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

x_train, x_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

x_train = np.reshape(x_train, (len(x_train), 3923, 57, 1))
x_test  = np.reshape(x_test,  (len(x_test),  3923, 57, 1))
print(x_train.shape)
print(x_test.shape)

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