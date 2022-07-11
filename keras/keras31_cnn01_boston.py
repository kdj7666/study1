# 12개 만들고 최적의 weight 가중치 파일을 저장할것 
from matplotlib import font_manager
from sklearn. datasets import load_boston  
import numpy as np 
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
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

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=55)

print(x_train.shape, x_test.shape) # (354, 13) (152, 13)
print(y_train.shape, y_test.shape) # (354,) (152,)

x_train= x_train.reshape(354,13,1,1)
x_test = x_test.reshape(152,13,1,1)

# print(x.shape)
# print(y.shape)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

# 2. 모델구성

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(1,1),
                 padding='same', input_shape=(13,1,1)))
model.add(Conv2D(16, (1,1), 
                 padding='valid',
                 activation='relu'))
model.add(Conv2D(8, (1,1), 
                 padding='valid',
                 activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

#3. 컴파일 , 훈련

start_time = time.time()

model.compile(loss='mae', optimizer='adam',
              metrics = ['mse'])

earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
              restore_best_weights=True)

a = model.fit(x_train, y_train, epochs=100, batch_size=10,
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
y_predict = np.argmax(y_predict, axis= 1)
y_predict = to_categorical(y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(x_test, y_predict)
print('r2스코어 : ', r2)

