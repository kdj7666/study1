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
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping



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




model = Sequential()
model.add(Dense(50, input_dim=4))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='swish'))
model.add(Dense(30, activation='swish'))
model.add(Dense(3, activation='softmax')) 



earlystopping = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit(x_train, y_train, epochs=1000, batch_size=1,
          validation_split=0.2,
          callbacks = [earlystopping],
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


y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc.score : ', acc)

# y_predict = np.argmax(y_predict, axis=1)
# y_test = np.argmax(y_test, reix= 1)
# print(y_test)
# print(y_predict)

# ----------------
# from sklearn.metrics import accuracy_score 
# acc = accuracy_score(y_test, y_predict)
# print('acc.score : ', acc)
# ---------------------

# req_index= np.array(1)
# np.argmax(y,out=req_index)
# print(req_index)


# def softmax(x) :
#         e_x = np.exp(x - np.max(y))

# req_index=np.array()
# req_index=np.argmax(y,axis=1)
# np.argmax(y,out=req_index)
# print(req_index)

# req_index=np.argmax(y,axis=0)
# print([req_index])


# x = [datasets]
# tf.argmax(x).eval(session=sess)

# y = np.array(datasets)
# np.argmax(y)



# print('x : ', x)
# print('softmax with -max : ', y, np.sum(y))


# def softmax(y) :
#         exp - np.exp(x - np.max(y))

#  print('softmax without -max : ', y)



# y_predict = model.predict(x_test[:5])















'''





import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#1.데이터
datasets = load_iris()
print(datasets.DESCR)
#4가지를 가르쳐줄테니 50개의 3종류 꽃으 맞춰라
# :Number of Instances: 150 (50 in each of three classes)
# # y=         - class:
#                 - Iris-Setosa
#                 - Iris-Versicolour
#                 - Iris-Virginica
print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
print(x)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=False)

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
model = Sequential()
model.add(Dense(5, input_dim=4))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax')) # y 값을 비율로 나눔 해서 아웃풋이 3개로 나눠야함 
                                          # 다중분류의 최종 노드의 갯수는 y의 갯수 

#3.컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=100, verbose=1, 
          validation_split=0.2,
          callbacks=ES)

#4.평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
from sklearn.metrics import accuracy_score

pre2 = y_predict.flatten() # 차원 펴주기
pre3 = np.where(pre2 > 1, 2 , 0) #1보다크면 2, 작으면 0
acc = accuracy_score(y_test, pre3)

print(pre3)
print('loss', loss)
print('acc', acc)

# loss [0.3140193819999695, 0.0]
# r2 0.0

# loss [0.46477609872817993, 0.0]
# r2 0.0

# 이중분류는 다이너리 크로스 엔트로피 만 쓴다 ( 당분간 )
# 다중분류는 카테코리 크로스 엔트로피 만 쓴다 ( 당분간 )
'''