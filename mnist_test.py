from tensorflow import keras
from tensorflow.keras.layers import Dense,Activation,Flatten
import numpy as np
import cv2

cont = 'y' # to continue the loop

model = keras.models.load_model("mnist_test.hdf5") #load the model
mnist = keras.datasets.mnist 
(train_inputs,train_targets),(test_inputs,test_targets) = mnist.load_data() 
normalised_train_inputs = train_inputs/255
normalise_test_inputs=test_inputs/255
normalise_test_inputs  = normalise_test_inputs.reshape(10000, 784)

prediction = model.predict_classes(normalise_test_inputs) #predict each image to coressponding number

# to check the given number is image predicted 
while(cont == 'y' or cont == 'Y'):
    test = int(input("enter the number to be tested 1 - 10000: "))
    print("predicted number----->",prediction[test-1])
    cv2.imshow("number_selected",test_inputs[test-1])
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    cont = input("to continue press y or Y:  ")


# To evaluate all 10000 numbers and getting its accuracy and loss
score = model.evaluate(normalise_test_inputs,test_targets)
print('Test loss:', score[0])  
print('Test accuracy:', score[1])

