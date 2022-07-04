from cProfile import label
import numpy as np 
from sklearn.datasets import load_iris
import tensorflow as tf
tf.random.set_seed(66)
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score 


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

# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

# from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder(sparse = False)
# ohe




y = to_categorical(y)
print(y)
print(y.shape) # ( 150 , 3 )

# ohe.fit(datasets.values.reshape(0, 1))


# print(type(ohe.categories_))

# ohe.categories_

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,
                    train_size=0.2,
                    shuffle=True, random_state=55) # shuffle True False 잘 써야 한다 
print(y_train)
print(y_test)


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(50, input_dim=4))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='swish'))
model.add(Dense(30, activation='swish'))
model.add(Dense(3, activation='softmax')) 

from tensorflow.python.keras.callbacks import EarlyStopping

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
























# loss :  0.07974613457918167
# accuracy :  0.9666666388511658

# weight 난수를 ( 4번으로 쓰겟다 ) 이렇게도 사용할수있다 위험하니 알아둘것 
# loss :  0.07870607078075409
# accuracy :  0.9750000238418579

