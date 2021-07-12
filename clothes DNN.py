import os
# remove warnings and info from debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from random import randrange

#load data
clothesData = keras.datasets.fashion_mnist

#split data into training + labels (all pics) and testing + labels (set of 10000 pics)
(trainingImages, trainingLabels), (testingImages, testingLabels) = clothesData.load_data()

#possible labels
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#normalize the inputs (0 - 1)
trainingImages = trainingImages / 255.0
testingImages = testingImages / 255.0

#constants for DNN
HIDDEN_LAYERS = 150
NUM_EPOCHS = 4

#create dense neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(HIDDEN_LAYERS, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

#train the DNN to the training set
model.fit(trainingImages, trainingLabels, epochs=NUM_EPOCHS)

loss, accuracy = model.evaluate(testingImages,  testingLabels, verbose=1)

print("Test accuracy:", accuracy)

#make an array of all predictions
predictions = model.predict(testingImages)

while True:
    #randomly display an image and show prediction vs expected
    predictionIndex = randrange(0, 10000)
    plt.figure()
    plt.title("Expected: " + labels[testingLabels[predictionIndex]] + "\nPrediction: " + labels[np.argmax(predictions[predictionIndex])])
    plt.imshow(testingImages[predictionIndex], cmap=plt.cm.binary)
    plt.show()
