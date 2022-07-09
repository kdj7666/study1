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
x_train = x[0:7]    # 성능이 데이터가 작고 완전히 정제된 데이터 이기에
x_test = x[0:7]       # 이렇게 할수있음 ( 이런식으론 절대 안한다) 성능 책임x
y_train = y[7:10]     # 1~10 연산이 오래걸려 30%를 자른다면 부분부분 자른다
y_test = y[7:10]      # 훈련 셋을 테스트 셋으로 옮길때 ( 16번 )
 
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


