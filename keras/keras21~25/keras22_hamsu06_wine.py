import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
tf.random.set_seed(66)
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score 
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
#1. data
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True))  # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))]


print(' y의 라벨값 :', np.unique(y)) # 판다스의 수치  y의 독특한게 무엇이냐 [ 0 1 2 ]

y = to_categorical(y)
print(y)
print(y.shape)

# ohe.fit(datasets.values.reshape(0, 1))


# print(type(ohe.categories_))

# ohe.categories_


x_train, x_test, y_train, y_test = train_test_split(x,y,
                    train_size=0.3,
                    shuffle=True, random_state=55) # shuffle True False 잘 써야 한다 
print(y_train)
print(y_test)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = RobustScaler()
scaler = MaxAbsScaler()

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))   # 0.0
print(np.max(x_train))   # 0.0 컬럼별로 나누어주어야 한다
print(np.min(x_test))
print(np.max(x_test))


# model 

# model = Sequential()
# model.add(Dense(50, input_dim=13))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(30, activation='swish'))
# model.add(Dense(30, activation='swish'))
# model.add(Dense(3, activation='softmax')) 

input1 = Input(shape=(13,))
dense1 = Dense(10)(input1)
dense2 = Dense(5, activation='relu')(dense1)
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(3, activation='softmax')(dense3)
model = Model(inputs=input1, outputs=output1)

model.save('./_save/k23_smm_wine.h5')


# compile , epochs 
earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

start_time = time.time()
a = model.fit(x_train, y_train, epochs=300, batch_size=100,
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
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc.score : ', acc)
print('걸린시간 : ', end_time)

model.summary()
# 없음 
# loss :  0.2520526349544525
# accuracy :  0.9120000004768372
# acc.score :  0.912
# 걸린시간 :  83.61341071128845

# min max 
# loss :  0.23565521836280823
# accuracy :  0.9599999785423279
# acc.score :  0.96
# 걸린시간 :  150.12877321243286

# standard
# loss :  0.6692562699317932
# accuracy :  0.9440000057220459
# acc.score :  0.944
# 걸린시간 :  387.00992608070374

# RobustScaler
# accuracy: 0.9520
# loss :  0.309433251619339
# accuracy :  0.9520000219345093
# acc.score :  0.952
# 걸린시간 :  9.102424383163452

# MaxAbsScaler
# accuracy: 0.8960
# loss :  0.4252156913280487
# accuracy :  0.8960000276565552
# acc.score :  0.896
# 걸린시간 :  9.75840139389038

# 함수형 모델 
# loss :  1.1018739938735962
# accuracy :  0.41600000858306885
# acc.score :  0.416
# 걸린시간 :  3.650966167449951
