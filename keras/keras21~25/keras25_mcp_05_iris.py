from cProfile import label
import numpy as np 
from sklearn.datasets import load_iris
import tensorflow as tf
tf.random.set_seed(66)
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score 
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. data

datasets = load_iris()
print(datasets.DESCR)       
        # Attribute Information: 4개의 컬럼 
        # - sepal length in cm
        # - sepal width in cm
        # - petal length in cm
        # - petal width in cm
        # #- class:
        #         - Iris-Setosa
        #         - Iris-Versicolour
        #         - Iris-Virginica      y 가 3개 
print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
print(x)
print(y)
print(x.shape, y.shape) # ( 150 , 4 ) ( 150 , )

print(' y의 라벨값 :', np.unique(y)) # 판다스의 수치  y의 독특한게 무엇이냐 [ 0 1 2 ]
print(np.unique(y, return_counts=True))


# from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder(sparse = False)
# ohe




y = to_categorical(y)
print(y)
print(y.shape) # ( 150 , 3 )

# ohe.fit(datasets.values.reshape(0, 1))


# print(type(ohe.categories_))

# ohe.categories_



x_train, x_test, y_train, y_test = train_test_split(x,y,
                    train_size=0.2,
                    shuffle=True, random_state=55) # shuffle True False 잘 써야 한다 
print(y_train)
print(y_test)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 0.0 컬럼별로 나누어주어야 한다
# print(np.min(x_test))
# print(np.max(x_test))


input1 = Input(shape=(4,))
dense1 = Dense(10)(input1)
dense2 = Dense(5, activation='relu')(dense1)
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(3, activation='softmax')(dense3)
model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# model.add(Dense(50, input_dim=4))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(30, activation='swish'))
# model.add(Dense(30, activation='swish'))
# model.add(Dense(3, activation='softmax')) 

# model.save('./_save/k23_smm_iris.h5')


earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1,
                              restore_best_weights=True)


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy', 'mse'])

start_time = time.time()

import datetime
date = datetime.datetime.now()
print(date)
date = date.strftime("%m%d_%H%M")
print(date)

filepath = './_ModelCheckPoint/k24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'


mcp = ModelCheckpoint(monitor='val_loss', node='auto', verbose=1,
                      save_best_only=True,
                      filepath= "".join([filepath, 'k24_iris', date, '_', filename]))

a = model.fit(x_train, y_train, epochs=300, batch_size=100,
          validation_split=0.2,
          callbacks = [earlystopping, mcp],
          verbose=1)

# loss ,acc = model.evaluate(x_test, y_test)

# print('loss : ', loss)
# print('accuracy : ', acc)

results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])

# print('====================================')
# print(y_test[:5])
# print('====================================')

# y_pred = model.predict(x_test[:5])
# print(y_pred)
print('====================================')
end_time = time.time()-start_time

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc.score : ', acc)
print('걸린시간 : ', end_time)
model.summary()

# 없음
# acc.score :  0.975
# 걸린시간 :  9.922988891601562

#min max 
# loss :  0.09696245193481445
# accuracy :  0.949999988079071
# acc.score :  0.95
# 걸린시간 :  14.008534669876099

# stardard
# loss :  4.395750045776367
# accuracy :  0.8166666626930237
# acc.score :  0.8166666666666667
# 걸린시간 :  117.38635349273682

# RobustScaler
# accuracy: 0.7917
# loss :  1.928321123123169
# accuracy :  0.7916666865348816
# acc.score :  0.7916666666666666
# 걸린시간 :  9.376346826553345

# MaxAbsScaler
# accuracy: 0.9667
# loss :  0.16355346143245697
# accuracy :  0.9666666388511658
# acc.score :  0.9666666666666667
# 걸린시간 :  8.586223602294922

# 함수형 모델 
# loss :  0.8824231028556824
# accuracy :  0.6333333253860474
# acc.score :  0.6333333333333333
# 걸린시간 :  10.695210695266724
