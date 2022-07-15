import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM

# 이해해라 무조건 
# 컬럼을 늘려도 봐라 
# range 함수는 마지막 숫자 -1 

a = np.array(range(1, 101)) # 1 ~ 100

size1 = 5  # x는 4개 y는 1개


def split_x(dataset, size1):
    aaa = []
    for i in range(len(dataset) - size1 + 1): # 10 - 9 + 1 = 2
        subset = dataset[i : (i + size1)]
        aaa.append(subset)  
    return np.array(aaa)

bbb = split_x(a, size1)
print(bbb)
print(bbb.shape) # (96, 5)
print('===========================================')
x = bbb[:, :-1]
y = bbb[:, -1]
print('===========================================')
print(x, y)
print(x.shape, y.shape) # (96, 4) (96,)
print('===========================================')

print('===========================================')

y_predict = qqq[:, -1]

print(x.shape, y.shape)

# 모델 구성 평가 예측할것 

model = Sequential()

model.add(Dense(64,input_shape=(4,1)))
model.add(Dense(61,activation='relu'))
model.add(Dense(37,activation='relu'))
model.add(Dense(54,activation='relu'))
model.add(Dense(86,activation='relu'))
model.add(Dense(17,activation='relu')) 
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=30 , batch_size=20, verbose=1)


loss = model.evaluate(x,y)
y_predict = y_predict.reshape(1,4,1)
result = model.predict(y_predict)

print('loss : 의 결과 ', loss)
print('[106]의 결과 : ', result)

