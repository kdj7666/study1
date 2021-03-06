import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. date
# x = np.array([1.2.3.4.5.6.7.8.9.10])
# y = np.array([1.2.3.4.5.6.7.8.9.10])      # 데이터 훈련은 전체를 다 해주고 그중 일부를 섞어 작업을 해준다 
x_train =np.array([1,2,3,4,5,6,7])          #train , test 분리해서 작업 해야 하고 간섭이 없어야 한다 
x_test = np.array([8,9,10])
y_train =np.array([1,2,3,4,5,6,7])
y_test = np.array([8,9,10])

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

#train , test 분리해서 작업 해야 하고 간섭이 없어야 한다 
