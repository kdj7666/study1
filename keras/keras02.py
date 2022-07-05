import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5]) 
# [실습] 맹그러봐!!!! 예측값은 6

model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=5)

loss = model.evaluate(x, y)
print("loss = ", loss)

result = model. predict([6])
print("예측값은 = ", result) 

