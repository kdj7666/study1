# 순차모델 seq
# 함수모델 

# add dense 
# layer 


# 위에서 정의
# layer
# layer
# layer
# layer
# layer
# layer
# layer
# 밑에서 정의 


import numpy as np 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],                        # y=wx + b   w=1, b=10
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
             [9,8,7,6,5,4,3,2,1,0]]     # 열의 갯수는 반드시 맞춘다 
             )                                                  #  (2,10) 을 (10,2)
y = np.array([11,12,13,14,15,16,17,18,19,20])
print(x.shape) # 10 , 3
print(y.shape) # 10 ,

# x = x.T            행열 변환 ( 14 15 16 )
# x = x.transpose()
# x = x.reshape(10,2)      <- 이것이 input_dim=2 특성  
x = x.T
print(x)
print(x.shape) # (10,3)


#2. 모델구성 순차형 모델
# model = Sequential()
# model.add(Dense (5, input_dim=3)) #  <- 특성이 3개   열 피쳐 컬럼 특성 mldel.add(Dense (5, input_dim=1)) <- 특성이 1개 
# model.add(Dense (5, input_shape=(3,)))
# model.add(Dense (4))
# model.add(Dense (3))
# model.add(Dense (2))
# model.add(Dense (2))
# model.add(Dense (1))


# 함수형 모델 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

input1 = Input(shape=(3,))
dense1 = Dense(10)(input1)
dense2 = Dense(5, activation='relu')(dense1) # 
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)

model.summary()

# Model: "model"    함수형 모델 
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 3)]               0
# _________________________________________________________________
# dense (Dense)                (None, 10)                40
# _________________________________________________________________
# dense_1 (Dense)              (None, 5)                 55
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)                 18
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 4
# =================================================================
# Total params: 117
# Trainable params: 117
# Non-trainable params: 0
# _________________________________________________________________



# 컴파일 아직 

