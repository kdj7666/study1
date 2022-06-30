from tkinter import Y
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np 

#1. data 
x = np.array(range(1,17))
y = np.array(range(1,17))
 
# [실습] 잘라봐

x_train = x[0:11] # 슬라이싱은 데이터 중복 가능 데이터값이명이 다르기 때문에 
y_train = y[0:11]

x_test = x[10:13]
y_test = y[10:13]

x_val = x[13:]
y_val = y[13:]

print(x_train)
print(x_train)

print(x_test)
print(y_test)

print(x_val)
print(y_val)


# x_train = np.array(range(1,11))  # range 끝에서 -1 해서 1 ~ 10 
# y_train = np.array(range(1,11))  # (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) [ 10 , ]
# x_test = np.array([11,12,13]) 
# y_test = np.array([11,12,13])   # evaluate = 
# x_val = np.array([14,15,16])    # validataion = 검증 
# y_val = np.array([14,15,16])    # 14 15 16 으로 검증을 하겠다  

# #2. model
'''
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))

# #3. compile 
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))

# #4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)



'''
