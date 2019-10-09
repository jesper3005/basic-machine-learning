import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

## Import of data from keras
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

## Makes data-set smaller
train_images = train_images/255.0
test_images = test_images/255.0

# print(train_images[7])

## Show image probably, cmap gives us the greyscale
# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()


# ---------------- MODEL -------------------------

## Sequential = A sequence of layer
## First input layer
## Two Dense layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# (data, data, epochs=how many times it iterates the data) Play with it for better accuracy
model.fit(train_images, train_labels, epochs=5)

## Evaluates the trained data
#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print("Tested acc: ", test_acc)

prediction = model.predict(test_images)

## Classification
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()

