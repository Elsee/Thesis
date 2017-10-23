# Larger CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import gzip
import six.moves.cPickle as pickle
import matplotlib, matplotlib.pyplot as plt

K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define the larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = larger_model()
# Fit the model
test = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

#loss_history = test.history["loss"]
#numpy_loss_history = numpy.array(loss_history)
#numpy.savetxt("./loss_history.txt", numpy_loss_history, delimiter=",")
#
#print(test.history.keys())
## summarize history 
#plt.plot(test.history['acc'], c='b', lw=1.5)
#plt.plot(test.history['val_acc'], c='r', lw=1.5)
#plt.plot(test.history['loss'], c='g', lw=1.5)
#plt.plot(test.history['val_loss'], c='m', lw=1.5)
#
#plt.title('model accuracy')
#plt.ylabel('loss/accuracy')
#plt.xlabel('epoch')
#plt.legend(['train acc', 'test acc', 'train loss', 'test loss'], loc='upper left')
#plt.tight_layout()
#plt.savefig('./result.jpg', format='jpg')
#plt.close()



encoded_imgs = model.predict(X_test)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cm

cm = confusion_matrix(y_test, encoded_imgs)
plt.matshow(cm)
plt.colorbar()
plt.show()

