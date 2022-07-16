import numpy as np
from sklearn.model_selection import train_test_split 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

# 1. 데이터

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])  # 13 , 3 
y = np.array((4,5,6,7,8,9,10,11,12,13,50,60,70))  # 13 , 

x_predict = np.array([50,60,70])    # 80 


print(x.shape, y.shape) # 13 , 3      13 , 

# input_shape = (행,열,몇개씩 자르는지!!!) rnn 에서 선생님이 만든것 
x = x.reshape(13, 3, 1) # x shape를 바꾼다 7, 3, 1 로 
print(x.shape)


# 2. model 
# rnn 모델 만드는 법 
model = Sequential()
# input_shape 는 행을 뺀다 ( 행 무시 열 우선 )                     input_dim
# model.add(SimpleRNN(26, input_shape=(3,1))) # [batch, timesteps, feature]
# model.add(SimpleRNN(10, input_length=3, input_dim=3)) # 이렇게도 쓸수 잇음 반대로 사용가능하지만 값이 달라짐 

# model.add(SimpleRNN(units=10, input_shape=(3,1)))
model.add(LSTM(units=10, input_shape=(3,1))) # 연산량이 많아진다 
# model.add(SimpleRNN(32))
# ValueError: Input 0 of layer simple_rnn_1 is incompatible with the layer: expected ndim=3, found ndim=2. Full 
# shape received: (None, 10) 
# (None, 64) 64개를 주었어 
model.add(Dense(100,activation='swish'))     # 윗줄 3차원 받을때는 2차원 ( 물어볼것 )
model.add(Dense(90,activation='swish'))
model.add(Dense(80,activation='swish'))
model.add(Dense(70,activation='swish'))
model.add(Dense(60,activation='swish'))
model.add(Dense(50,activation='swish'))
model.add(Dense(1))

model.summary()
# [simple] units : 10 ->  10 * (1 + 1 + 10 ) = 120
# [LSTM] units : 10 -> 4 * 10 * (1 + 1 + 10) = 480
                        # 4*20*(1+1+20) = 1760
# 결론  : LSTM = simpleRNN * 4
# 숫자 4의 의민는 call state, input gate, output gate, forget gate 


# 3. cmopile , epochs
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs= 850, batch_size=20, verbose=1)

# 4. evaulate , predict
loss = model.evaluate(x,y)
y_pred = np.array([50,60,70]).reshape(1,3,1) #   [[[8], [9], [10]]]                                ([8,9,10]) 1차원이기에 reshape로 차원을 바꿔준다 
result = model.predict(y_pred)

print('loss : 의 결과 ', loss)
print('80의 결과 : ', result)


