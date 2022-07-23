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

x_predict = np.array(range(96,106)) # 96 ~ 105 

size2 = 7
def split_g(dataset, size2):
    rrr = []
    for l in range(len(dataset) - size2 + 1): # 10 - 7 + 1 = 4
        subset = dataset[l : (l + size2)]
        rrr.append(subset)  
    return np.array(rrr)


bbb = split_x(a, size1)
print(bbb)
print(bbb.shape) # (96, 5)
print('===========================================')
x = bbb[:, :-1]
y = bbb[:, -3]
print('===========================================')
print(x, y)
print(x.shape, y.shape) # (96, 4) (96,)
print('===========================================')

qqq = split_g(x_predict, size2)
print(qqq) # 7,4

print('===========================================')

y_predict = qqq[:, -3]

print(y_predict)

x = x.reshape(96, 4, 1)
print(x.shape)



# 모델 구성 평가 예측할것 

model = Sequential()

model.add(LSTM(units=100, input_shape=(4,1)))
model.add(Dense(100, activation='swish'))
model.add(Dense(80, activation='swish'))
model.add(Dense(60, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=60 , batch_size=20, verbose=1)


loss = model.evaluate(x,y)
y_predict = np.array(y_predict).reshape(1,4,1)
result = model.predict(y_predict)

print('loss : 의 결과 ', loss)
print('106이 나와야함 : ', result)

