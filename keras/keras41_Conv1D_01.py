# TimeSeriese data 시계열 데이터 ( 주가 )라고 생각하면 편함 ( 3차원 )
# dnn 2 cnn 4 rnn 3     rnn 은 자르는 단위를 씀 
# 시계열 데이터란 일정한 시간동안 수집 된 일련의 순차적으로 정해진 데이터 셋의 집합 입니다. 
# 시계열 데이터의 특징으로는 시간에 관해 순서가 매겨져 있다는 점과,
# 연속한 관측치는 서로 상관관계를 갖고 있습니다.
# 통상 시계열 데이터는 y값이 없기에  x , y 를 분리해줘야한다 
# 순서가 그 다음 순서에 영향을 끼친다 
# 연속된 data를 자른 후에 rnn모델에 맞춰  shape 
# rnn ( Recurrent Neural Network ) Recurrent ( 반복적인 순환하는)


# 1 2 3 4 5 6 7 8 9 
# 123을 잘라서 4의 값을 예측 ( 6번)

#  1. data 

# datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# size = 5 

# def split_x(seq, size):
# -----------------------------------------------------------------------

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Conv1D, Flatten
# from tensorflow.keras.layers import Bidirectional
from sklearn.metrics import r2_score


# 1. data 

datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]]) # 7 , 3
y = np.array([4,5,6,7,8,9,10]) # 7 ,

print(x.shape, y.shape)

# input_shape = (행,열,몇개씩 자르는지!!!) rnn 에서 선생님이 만든것 
x = x.reshape(7, 3) # x shape를 바꾼다 7, 3, 1 로 
print(x.shape)

# 2. model 
# rnn 모델 만드는 법 
model = Sequential()
# input_shape 는 행을 뺀다 ( 행 무시 열 우선 )
# model.add(LSTM(10, input_shape=(3,1), return_sequences=False))
model.add(Conv1D(10, 2, input_shape=(3,1)))
model.add(Flatten())
model.add(Dense(1, activation='relu'))
model.summary() # LSTM : S17 // Conv1D : 97


# 3. cmopile , epochs
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs= 400, verbose=1)

# 4. evaulate , predict
loss = model.evaluate(x,y)
#   [[[8], [9], [10]]]                                ([8,9,10]) 1차원이기에 reshape로 차원을 바꿔준다 


print('loss : 의 결과 ', loss)
print('[8,9,10의 결과 : ', result)

# [8,9,10의 결과 :  [[10.854522]]

# [8,9,10의 결과 :  [[10.8882065]]

# [8,9,10의 결과 :  [[10.890814]]

# [8,9,10의 결과 :  [[10.918358]]

# [8,9,10의 결과 :  [[10.920321]]


