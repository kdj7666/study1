# TimeSeriese data 시계열 데이터 ( 주가 )라고 생각하면 편함 ( 3차원 )
# dnn 2 cnn 4 rnn 3     rnn 은 자르는 단위를 씀 
# 시계열 데이터란 일정한 시간동안 수집 된 일련의 순차적으로 정해진 데이터 셋의 집합 입니다. 
# 시계열 데이터의 특징으로는 시간에 관해 순서가 매겨져 있다는 점과,
# 연속한 관측치는 서로 상관관계를 갖고 있습니다.
# 통상 시계열 데이터는 y값이 없기에  x , y 를 분리해줘야한다 
# 순서가 그 다음 순서에 영향을 끼친다 
# 연속된 data를 자른 후에 rnn모델에 맞춰  shape 

# LSTM 4가지 정리 후 이해 
# LSTM 정리 후LSTM2.png 선생님에게 보내기 
# LSTM 그림 모델 줄마다 설명 달기 

# 1 2 3 4 5 6 7 8 9 
# 123을 잘라서 4의 값을 예측 ( 6번)

#  1. data 

# datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# size = 5 

# def split_x(seq, size):
# -----------------------------------------------------------------------
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM, GRU


# 1. data 

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])  # 13 , 3 
y = np.array((4,5,6,7,8,9,10,11,12,13,50,60,70))  # 13 , 

x_predict = np.array([50,60,70])    # 80  /  3차원 

print(x.shape, y.shape)

# input_shape = (행,열,몇개씩 자르는지!!!) rnn 에서 선생님이 만든것 
x = x.reshape(13, 3, 1) # x shape를 바꾼다 7, 3, 1 로 
print(x.shape)

# 2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(10, return_sequences=True, input_shape=(3,1))) # return_sequences=True ( 차원자체가 1이 늘어남 ) 3차원  ( n , 3 , 1 ) -> ( n , 3 , 10 ) cnn 피쳐가 바뀜  
model.add(LSTM(5, return_sequences=False)) # 
model.add(Dense(1))

model.summary()

# istm 2개 엮은거 테스트해보고 

