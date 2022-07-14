# 12개 만들고 최적의 weight 가중치 파일을 저장할것 
from matplotlib import font_manager
from sklearn. datasets import load_boston  
import numpy as np 
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터

datasets = load_boston()
x, y = datasets.data, datasets.target

print(x.shape)
print(y.shape)

x = x.reshape(506,13,1)
print(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=55)



# print(np.unique(y_train, return_counts=True))
# print(np.unique(y_test, return_counts=True))

# 2. 모델구성
# cnn ( 4 차원 ) # lstm = rnn 3차원 reshape 로 3차원 변환
model = Sequential()

model.add(LSTM(units=64, return_sequences=True,
               input_shape=(13,1)))
model.add(LSTM(32, return_sequences=False,
               activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

#3. 컴파일 , 훈련

start_time = time.time()

model.compile(loss='mae', optimizer='adam',
              metrics = ['mse'])

earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
              restore_best_weights=True)

a = model.fit(x_train, y_train, epochs=100, batch_size=100,
                validation_split=0.2,
                callbacks=[earlystopping],
                verbose=1)

print(a)
print(a.history['val_loss'])

end_time = time.time() - start_time

# 4 . 평가 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# print('결과값 : ', result)

# ValueError: cannot reshape array of size 4602 into shape (1,3,1) 리쉐이프가 필요없다 
# 데이터 전처리에서 3차원으로 변환 후 모델 입력값을 넣었다 이후 모델 Dense를 통해 2차원으로 나옴 마지막에 다시 리쉐이프가 필요없음 

# r2스코어 :  -6.42992388350632

# r2스코어 :  -0.3617267790998777

# r2스코어 :  0.6736686360098978

