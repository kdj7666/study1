from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split

#1. data 
x = np.array(range(1,17))
y = np.array(range(1,17))
# https://mambo-coding-note.tistory.com/186 필독 / 연습할것 
x_train, x_test , y_train , y_test = train_test_split(x,y,
    test_size=0.2, random_state=66
)
print(x_train.shape, x_test.shape) 

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
          validation_split=0.25) # train date 를 25프로 사용하겠다 

# #4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)

