
from tensorflow import keras
from tensorflow.keras.layers import Dense,Activation,Flatten
import numpy as np
import cv2

mnist = keras.datasets.mnist #
(train_inputs,train_targets),(test_inputs,test_targets) = mnist.load_data() #getting or loading mnist data set into the variable

# normalisation for faster gradient decent calculation
normalised_train_inputs = train_inputs/255
normalise_test_inputs=test_inputs/255

#creating a neural network
model=keras.Sequential()
model.add(Flatten(input_shape=normalised_train_inputs.shape[1:]))
model.add(Dense(128,input_shape = normalised_train_inputs.shape[1:])) 						
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax')) 									
model.compile(loss="sparse_categorical_crossentropy",optimizer=keras.optimizers.Adam(learning_rate=0.01),metrics=['accuracy'])
model.fit(normalised_train_inputs,train_targets,batch_size=128,epochs=20)

#save model
model.save("mnist_test.hdf5")
