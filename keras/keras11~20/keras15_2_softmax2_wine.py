import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
tf.random.set_seed(66)
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score 

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


# model 

model = Sequential()
model.add(Dense(50, input_dim=13))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='swish'))
model.add(Dense(30, activation='swish'))
model.add(Dense(3, activation='softmax')) 

# compile , epochs 
earlystopping = EarlyStopping(monitor='val_loss', patience=150, mode='auto', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit(x_train, y_train, epochs=2000, batch_size=1,
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


y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc.score : ', acc)


# print('loss : ', loss)
# print('accuracy : ', acc)

# accuracy: 0.8800
# loss :  0.3663438856601715
# accuracy :  0.8799999952316284
# ====================================