from tkinter import Y
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split
#1. data 
x = np.array(range(1,17))
y = np.array(range(1,17))

# [실습] train_test_split로만 나누어라 10 : 3 : 3

x_train, x_test , y_train, y_test = train_test_split(x,y, test_size=0.625, random_state=1) # 이 함수로는 두쌍만 가능 다시 복습할것 

x_val, y_val , x_fdfd , y_fdfd = train_test_split(x_train, y_train, train_size=0.5 , random_state=12)
print(x_train)
print(x_test)
print(x_val)

# x_train = np.array(range(1,11))  # range 끝에서 -1 해서 1 ~ 10 
# y_train = np.array(range(1,11))  # (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) [ 10 , ]
# x_test = np.array([11,12,13]) 
# y_test = np.array([11,12,13])   # evaluate = 
# x_val = np.array([14,15,16])    # validataion = 검증 
# y_val = np.array([14,15,16])    # 14 15 16 으로 검증을 하겠다  

# #2. model

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



