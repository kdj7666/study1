# import numpy as np
from sklearn.datasets import load_digits
# -------------------------------
from cProfile import label
import numpy as np 
import tensorflow as tf
tf.random.set_seed(66)
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, r2_score 
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# ---------------------------------
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. data

datasets = load_digits()
x = datasets.data
y = datasets.target


print(x.shape)
x = x.reshape(1797,64,1)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                    train_size=0.7,
                    shuffle=True, random_state=65) # shuffle True False 잘 써야 한다 

print(x_train.shape, x_test.shape) # (359,64) (1438,64)
print(y_train.shape, y_test.shape) # (359,) (1438,)


print(np.unique(y_train, return_counts=True))  # [0 1 2 3 4 5 6 7 8 9]  10 개 
print(np.unique(y_test, return_counts=True))  # [0 1 2 3 4 5 6 7 8 9]  10 개 

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[0])
# plt.show()


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


# # model 

model = Sequential()
model.add(LSTM(units=64, return_sequences=True,
               input_shape=(64,1)))
model.add(LSTM(32, return_sequences=False,
               activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


# input1 = Input(shape=(64,))
# dense1 = Dense(10)(input1)
# dense2 = Dense(5, activation='relu')(dense1)
# drop1 = Dropout(0.4)(dense2)
# dense3 = Dense(3, activation='relu')(drop1)
# output1 = Dense(10, activation='softmax')(dense3)
# model = Model(inputs=input1, outputs=output1)


# compile , epochs 
start_time = time.time()
earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'mse', optimizer = 'adam',
              metrics = ['accuracy', 'mse'])


a = model.fit(x_train, y_train, epochs=100, batch_size=3000,
          validation_split=0.2,
          callbacks = [earlystopping],
          verbose=1)

end_time = time.time()-start_time
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

y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)
# y_test = np.argmax(y_test, axis=1)
# print(y_test)
r2 = r2_score(y_test, y_predict)
# acc = accuracy_score(y_test, y_predict)
# print('acc.score : ', acc)
print('걸린시간 : ', end_time)
print('r2.score:', r2)
model.summary()

# acc.score :  0.9814814814814815
# 걸린시간 :  19.17873787879944
# r2.score: 0.9758885412737569

# loss :  5.227468490600586
# accuracy :  0.10185185074806213
# ====================================
# 걸린시간 :  97.58717966079712
# r2.score: 0.3517845470062344

