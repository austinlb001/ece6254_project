# AUTHOR(S): BARBER, AUSTIN L.; FEIGHT, JORDAN; KHANDI, ROHAN; KHANPOUR, CAMERON; NELSON, ALEXIS
# FILE NAME: ece6254_project_cnn.py 
# DATE CREATED: 2024 APRIL 09
# DATE LAST UPDATED: 2024 APRIL 09
# PURPOSE: THIS FILE IS THE MAIN FILE FOR THE CONVOLUTION NEURAL NETWORK PORTION OF THE PROJECT.

############### I M P O R T S ###############
import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt



############### M A I N ###############
if __name__ == "__main__":

    # VARIABLES
    D1 = 64 # NUMBER OF NODES IN THE FIRST DENSE LAYER
    D2 = 16 # NUMBER OF NODES IN THE SECOND DENSE LAYER
    d3 = 1 # NUMBER OF NODES IN THE THRID DENSE LAYER
    dropout_rate = 0.2 # THE DROPOUT RATE
    train_epochs = 4 # NUMBER OF EPOCHS FOR TRAINING    

    # LOAD DATASET
    (xtrain, ytrain), (xtest, ytest) = ???

    # BUILD THE MODEL
    model = keras.Sequential()
    model.add(keras.layers.Conv3D((2, 80), (1, 10, 1))) # INPUT CONVOLUTION LAYER
    model.add(keras.layers.Dropout(dropout_rate)) # DROPOUT LAYER
    model.add(keras.layers.Dense(D1, avtivation = 'tanh'), kernel_regulizer = 'l2') # LAYER 3
    model.add(keras.layers.Dense(D2, avtivation = 'tanh'), kernel_regulizer = 'l2') # LAYER 4
    model.add(keras.layers.Dense(D3, avtivation = 'tanh'), kernel_regulizer = 'l2') # OUTPUT LAYER

    # COMPILE THE MODEL
    model.compile(optimizer = 'SGD', loss = ???, metrics = 'accuracy')


    # TRAIN THE MODEL
    history = model.fit(xtrain, ytrain, train_epochs = train_epochs, validation_data = (xval, yval))

    # EVALUATE THE MODEL
    test_loss, test_acc = model.evaluate(xtest, ytest, verbose = 2)
    print("Test Accuracy = ", test_acc)
    