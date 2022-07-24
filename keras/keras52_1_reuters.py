from keras.datasets import reuters
import numpy as np
import pandas as pd 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Flatten, Embedding, Reshape
# 1. data

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000)

print(x_train.shape)             # ( 8982 , ) 리스트가 8982개 이다 
print(x_test.shape)              # ( 2246 , ) 
print(y_train,y_train.shape)     # [ 3  4  3 ... 25  3 25] (8982,)
print(y_test,y_test.shape)       # [ 3 10  1 ...  3  3 24] (2246,)
print(np.unique(y_train, return_counts=True)) # 46 개의 뉴스카테고리 
print(len(np.unique(y_train)))   # 46

print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0]))             # <class 'list'> 페드시쿼트

# print(x_train[0].shape) # AttributeError: 'list' object has no attribute 'shape'

print(len(x_train[0]))              # 87
print(len(x_train[1]))              # 56 다 같게 해주어야한다 

print('뉴스기사의 최대길이 : ', max (len(i) for i in x_train))  # 2376   8982개 중에 제일 긴 길이를 내어준다
print('뉴스기사의 평균길이 : ', sum(map(len, x_train)) / len(x_train)) # 145.53  각각의 길이의 평균을 내어준다

# 전처리 

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre') # 8982, -> 8982, 100
# 마지막이 자르는것 나중의 데이터가 더 중요할수있으니 앞에서 부터 자른다 [다시 찾아볼것] 
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre') 

# x_train = to_categorical(y_train)
# x_test = to_categorical(y_test)

print(x_train.shape, y_train.shape)   # (8982, 100) (8982,)
print(x_test.shape, y_test.shape)     # (2246, 100) (2246,)

x_train = x_train.reshape(8982,100,1)
x_test = x_test.reshape(2246,100,1)

#2. model 

model = Sequential()     
model.add(Embedding(input_dim=10000, output_dim=100))
model.add(LSTM(32))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(46, activation='softmax'))
model.summary()

# 3. compile , epochs
import time
start_time = time.time()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=650, batch_size=600)
end_time = time.time() - start_time
# 4. evaluate , predict

acc = model.evaluate(x_test, y_test)
print('acc : ', acc)
print('걸린시간 : ', end_time)



# acc :  [4.469789028167725, 0.6242208480834961]
# 걸린시간 :  6332.253529548645



# acc :  [4.615542888641357, 0.6317898631095886]
# 걸린시간 :  5112.622207403183

# acc :  [13.58715534210205, 0.5565449595451355]
# 걸린시간 :  5072.371233463287
