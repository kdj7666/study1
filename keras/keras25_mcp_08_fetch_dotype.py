# [과제] 만들어서 속도 비교 
# gpu 와 cpu 

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
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) :
    print('돈다')
    aaa = 'gpu'
else : 
    print('안돈다')
    aaa = 'cpu'


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


# model 

# model = Sequential()
# model.add(Dense(60, input_dim=54))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(50, activation='swish'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(7, activation='softmax')) 


input1 = Input(shape=(54,))
dense1 = Dense(10)(input1)
dense2 = Dense(5, activation='relu')(dense1)
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(7, activation='softmax')(dense3)
model = Model(inputs=input1, outputs=output1)

# model.save('./_save/k23_smm_fetch_dotyupe.h5')



# compile , epochs 


start_time = time.time()

import datetime
date = datetime.datetime.now()
print(date)
date = date.strftime("%m%d_%H%M")
print(date)

filepath = './_ModelCheckPoint/k24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlystopping = EarlyStopping(monitor='val_loss', patience=150, mode='auto', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])



mcp = ModelCheckpoint(monitor='val_loss', node='auto', verbose=1,
                      save_best_only=True,
                      filepath= "".join([filepath, 'k24_fetch_dotype', date, '_', filename]))
a = model.fit(x_train, y_train, epochs=100, batch_size=320,
          validation_split=0.2,
          callbacks = [earlystopping, mcp],verbose=1)

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




end_time = time.time()-start_time
from sklearn.metrics import confusion_matrix

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
# y_test = np.argmax(y_test, axis=1)
print(y_test)
print(aaa, ' 걸린시간 : ', end_time)
acc = accuracy_score(y_test, y_predict)
print('acc.score : ', acc)

model.summary()

# 없음 
# loss :  0.42078331112861633
# accuracy :  0.8222241401672363
# gpu  걸린시간 :  218.40483736991882

# min max 
# loss :  0.316653311252594
# accuracy :  0.8728147149085999
# gpu  걸린시간 :  218.75451636314392

# standard 
# loss :  0.298806369304657
# accuracy :  0.8857748508453369
# gpu  걸린시간 :  218.64443588256836

# 함수형 모델 
# loss :  1.2059673070907593
# accuracy :  0.48737117648124695
# gpu  걸린시간 :  136.89638090133667
