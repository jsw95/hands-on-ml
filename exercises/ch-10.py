import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

fashion_data = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_data.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(units=300, activation="relu"))
model.add(keras.layers.Dense(units=100, activation="relu"))
model.add(keras.layers.Dense(units=len(class_names), activation="softmax"))

print(model.summary())

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=0.01),
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

X_new = X_test[0]
y_pred = model.predict_classes(X_new.reshape(-1, 28, 28))

prediction = class_names[y_pred]
print(prediction)

plt.imshow(X_new)
plt.show()

# success