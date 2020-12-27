
import keras
import numpy as np
from matplotlib import pyplot as plt


def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)

n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
# X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

y_pred = X_valid[:, -1]
np.mean(keras.losses.mean_squared_error(y_valid, y_pred))


_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[50, 1]),
    keras.layers.Dense(1)
])
model = keras.models.Sequential([
  keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
  keras.layers.SimpleRNN(20),
  keras.layers.Dense(1)
])
print(model.summary())

model.compile(loss="mean_squared_error",
              optimizer=keras.optimizers.SGD(learning_rate=0.01),
              metrics=["accuracy"])

# model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))


# predicting n steps forward
series = generate_time_series(10000, n_steps + 10)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(10)
])

model.compile(loss="mean_squared_error",
              # optimizer=keras.optimizers.SGD(learning_rate=0.01),
              metrics=["accuracy"])

model.fit(X_train, Y_train, epochs=10, validation_data=(X_valid, Y_valid))
y_pred = model.predict(X_valid[:, 10:])[:,np.newaxis, :]


plt.plot(np.concatenate([X_valid[3].reshape(1, -1)[0], Y_valid[3]]), color="red")
plt.plot(np.concatenate([X_valid[3].reshape(1, -1)[0], y_pred[3][0]]), color="green")
plt.plot(X_valid[3])
plt.show()
