import tensorflow as tf


mnist = tf.keras.datasets.mnist
(x_train,y_train), (x_test, y_test) = mnist.load_data()

# x_train is 60,000 X28X28
# y_train is
y_train, y_test = tf.cast(y_train,tf.int64),tf.cast(y_test,tf.int64)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation="relu")) # layer-1
model.add(tf.keras.layers.Dense(128, activation="relu")) # layer-2
model.add(tf.keras.layers.Dense(10, activation="softmax")) # layer-3 -- output layer

# Optimizer is sigmoid
# loss functions is sparse categorical crossentropy
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10)
result = model.evaluate(x_test, y_test)

print(result)






