from matplotlib import font_manager
from sklearn. datasets import load_boston  
import numpy as np 
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

# 1. 데이터

datasets = load_boston()  
x, y = datasets.data, datasets.target       

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=55)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델구성

model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()


#3. 컴파일 , 훈련


model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
              restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss', node='auto', verbose=1,
                      save_best_only=True,                                        # save_best_only = true 가장 좋은 것을 맞다 false 아니다
                      filepath='./_ModelCheckPoint/keras24_ModelCheckpoint.hdf5') # filepath 파일에 넣다  '여기' 

start_time = time.time()

a = model.fit(x_train, y_train, epochs=1000, batch_size=1,
                validation_split=0.25,
                callbacks=[earlystopping, mcp],
                verbose=1)


# print(a)
# print(a.history['loss']) # 대괄호로 loss , val loss 값 출력 가능

# end_time = time.time() - start_time

# 4 . 평가 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2_score : ', r2)
# print('걸린시간 : ', end_time)


# loss :  12.638511657714844
# r2_score :  0.8254214332059434

