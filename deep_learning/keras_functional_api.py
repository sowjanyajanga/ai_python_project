import tensorflow
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input


input = Input(shape=(2,))

layer1 = Dense(10, activation='relu')(input)
layer2 = Dense(10, activation='relu')(layer1)
output = Dense(1, activation='sigmoid')(layer2)

model = Model(inputs=input, outputs=output)

# optimizer for weights is gradident descent
# loss function binary cross entropy
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])



