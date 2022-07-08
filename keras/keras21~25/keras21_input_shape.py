import numpy as np 
from tensorflow.python.keras.models import Sequential      
from tensorflow.python.keras.layers import Dense     

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],                        # y=wx + b   w=1, b=10
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
             [9,8,7,6,5,4,3,2,1,0]]     # 열의 갯수는 반드시 맞춘다 
             )                                                  #  (2,10) 을 (10,2)
y = np.array([11,12,13,14,15,16,17,18,19,20])
print(x.shape) # (2,10)
print(y.shape) # (10,) -> (10,1) 이라 할수 있다

# x = x.T            행열 변환 ( 14 15 16 )
# x = x.transpose()
# x = x.reshape(10,2)      <- 이것이 input_dim=2 특성  
x = x.T
print(x)
print(x.shape) # (10,2)


#2. 모델구성
model = Sequential()
# model.add(Dense (5, input_dim=3)) #  <- 특성이 2개   열 피쳐 컬럼 특성 mldel.add(Dense (5, input_dim=1)) <- 특성이 1개 
model.add(Dense (5, input_shape=(3,)))
model.add(Dense (4))
model.add(Dense (3))
model.add(Dense (2))
model.add(Dense (2))
model.add(Dense (1))

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 5)                 20

#  dense_1 (Dense)             (None, 4)                 24

#  dense_2 (Dense)             (None, 3)                 15

#  dense_3 (Dense)             (None, 2)                 8

#  dense_4 (Dense)             (None, 2)                 6

#  dense_5 (Dense)             (None, 1)                 3

# =================================================================
# Total params: 76
# Trainable params: 76
# Non-trainable params: 0
# _________________________________________________________________

