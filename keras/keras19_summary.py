from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. data
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. model
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.summary() # output shape 에서 b 가 붙음 

#3. compile , epochs 

