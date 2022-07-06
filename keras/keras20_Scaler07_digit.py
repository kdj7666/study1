# import numpy as np
from sklearn.datasets import load_digits
# -------------------------------
from cProfile import label
import numpy as np 
import tensorflow as tf
tf.random.set_seed(66)
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score 
from tensorflow.python.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
# ---------------------------------
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
#1. data
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)   #  (1797, 64) (1797,)
print(np.unique(y, return_counts=True))  # [0 1 2 3 4 5 6 7 8 9]  10 개 

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[0])
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x,y,
                    train_size=0.2,
                    shuffle=True, random_state=65) # shuffle True False 잘 써야 한다 
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

model = Sequential()
model.add(Dense(50, input_dim=64))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='swish'))
model.add(Dense(30, activation='swish'))
model.add(Dense(10, activation='softmax')) 


# compile , epochs 
start_time = time.time()
earlystopping = EarlyStopping(monitor='val_loss', patience=150, mode='auto', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit(x_train, y_train, epochs=2500, batch_size=80,
          validation_split=0.2,
          callbacks = [earlystopping],
          verbose=1)

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
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
# y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc.score : ', acc)
print('걸린시간 : ', end_time)

# 없음 
# loss :  0.3793579339981079
# accuracy :  0.8963838815689087
# acc.score :  0.8963838664812239
# 걸린시간 :  8.7983877658844

# min max 
# loss :  0.3464629352092743
# accuracy :  0.9033379554748535
# acc.score :  0.9033379694019471
# 걸린시간 :  9.165494918823242

# standard
# loss :  0.41743403673171997
# accuracy :  0.8873435258865356
# acc.score :  0.8873435326842837
# 걸린시간 :  7.543145895004272

