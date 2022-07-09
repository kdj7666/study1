# import numpy as np
from json import encoder
from sklearn.datasets import fetch_covtype
# ---------------------------------
from cProfile import label
import numpy as np 
import tensorflow as tf
tf.random.set_seed(66)
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score 
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------
import pandas as pd
import tensorflow as tf
# ---------------------------------------------------


#1. data
datasets = fetch_covtype()
x = datasets.data
y = datasets.target.reshape(-1,1)

print(x.shape, y.shape) #  (581012, 54) (581012,)
print(np.unique(y, return_counts=True))  # [ 1 2 3 4 5 6 7]

print(datasets.feature_names)

encoder = OneHotEncoder()
encoder.fit(y)
y = encoder.transform(y).toarray()
print(y.shape)
print(y)

# ------------------------------------------

# pd.get_dummies(fetch_covtype)
# print(datasets.feature_names)

# onhot_encoder = OneHotEncoder()
# minmax_scaler = MinMaxScaler()
# fetch_covtype = minmax_scaler.fit_transform

# print(fetch_covtype[:10])
# ---------------------------------------------


x_train, x_test, y_train, y_test = train_test_split(x,y,
                    train_size=0.2,
                    shuffle=True, random_state=44) # shuffle True False 잘 써야 한다 

print(y_train)
print(y_test)


# model 

model = Sequential()
model.add(Dense(108, input_dim=54))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='swish'))
model.add(Dense(100, activation='swish'))
model.add(Dense(7, activation='softmax')) 

model.summary()
# compile , epochs 
earlystopping = EarlyStopping(monitor='val_loss', patience=150, mode='auto', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit(x_train, y_train, epochs=300, batch_size=3200,
          validation_split=0.2,
          callbacks = [earlystopping],verbose=1)

# loss ,acc = model.evaluate(x_test, y_test)

results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])

# print('====================================')
# print(y_test[:5])
# print('====================================')


# y_pred = model.predict(x_test[:5])
# print(y_pred)
print('====================================')

from sklearn.metrics import confusion_matrix

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
# y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc.score : ', acc)


# print('loss : ', loss)
# print('accuracy : ', acc)


# tensorflow one hot encoding
# loss :  0.6355554461479187
# accuracy :  0.7333232760429382

# minmaxscaler  scaler -> 85% 이상이면 좋은 지표  

# 데이터 전처리에 scaler 사용할것이다 
# 모든 x 의 값을 0과 1사이로 바꿀것이고
# x에서 y에 가는 길이 틀어지면 조작
# y의 값이 달라지면 조작
# x , y 값이 늘어나거나 줄어들면 조작 
# y값이 셔플하거나 저기 했을때 데이터가 달라지면 조작 
