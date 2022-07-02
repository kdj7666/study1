import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. date
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# x_train =np.array([1,2,3,4,5,6,7])
# x_test = np.array([8,9,10])
# y_train =np.array([1,2,3,4,5,6,7])
# y_test = np.array([8,9,10])

# [과제] 넘파이 리스트의 슬라이싱!! 7:3으로 잘라라
x_train = x[0:7]     # 0 : 7 = 0 ~ 7-1 (6)
x_test = x[0:7]        #    공백으로 처음부터 가능 , 끝가지 가능 
y_train = y[7:10]
y_test = y[7:10]

print(x_train)
print(x_test)
print(y_train)
print(y_test)

#2. model             딥러닝 인풋1개 아웃풋1개 히든레이어 10개
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

#3. 컴파일 , 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300 , batch_size=1)

#4. 평가 예측                                            # 훈련은 다 시켜주면 좋다 
loss = model.evaluate(x_test, y_test)
print('loss', loss)
result = model.predict([11])
print('11의 예측값 : ',result)
