from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

# First layer from input
# 13 is the size of your fully connected layer
# 7 is the dimension of your input
# relu is the activation function
model.add(Dense(13, input_dim=7, activation='relu'))

#Next layer(hidden)
model.add(Dense(7, activation='relu'))

# Output layer with one output
model.add(Dense(1, activation='sigmoid'))
