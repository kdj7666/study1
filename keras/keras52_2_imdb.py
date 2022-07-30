from keras.datasets import imdb
from matplotlib.cbook import flatten
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000) # 디폴트로잡힌다 num_wodrs는 단어의 반도수 설정, embedding의 input_dim에 넣어주면 됨 

print(x_train.shape)             # ( 25000 , ) 리스트가 25000개 이다 
print(x_test.shape)              # ( 25000 , ) 
print(y_train,y_train.shape)     # [1 0 0 ... 0 1 0] (25000,)
print(y_test,y_test.shape)       # [0 1 1 ... 0 0 0] (25000,)
print(np.unique(y_train, return_counts=True)) # (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
print(len(np.unique(y_train)))   # 2

print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

print(type(x_train[0]))             # <class 'list'> 페드시쿼트

# # print(x_train[0].shape) # AttributeError: 'list' object has no attribute 'shape'

print(len(x_train[0]))              # 218
print(len(x_train[1]))              # 189 다 같게 해주어야한다 

print('아이엠디비의 최대길이 : ', max (len(i) for i in x_train))  # 2494   25000개 중에 제일 긴 길이를 내어준다
print('아이엠디비의 평균길이 : ', sum(map(len, x_train)) / len(x_train)) # 238.71364  각각의 길이의 평균을 내어준다



# 전처리 

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, LSTM, Flatten, Embedding
x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre') # 8982, -> 8982, 100
# 마지막이 자르는것 나중의 데이터가 더 중요할수있으니 앞에서 부터 자른다 [다시 찾아볼것] 
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre') 

# x_train = to_categorical(y_train)
# x_test = to_categorical(y_test)

print(x_train.shape, y_train.shape)   # (25000, 100) (25000,)
print(x_test.shape, y_test.shape)     # (25000, 100) (25000,)

x_train = x_train.reshape(25000,100,1)
x_test = x_test.reshape(25000,100,1)

#2. model 

model = Sequential()     
model.add(Embedding(input_dim=10000, output_dim=100))
model.add(LSTM(32))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax'))
model.summary()

# 3. compile , epochs
import time
start_time = time.time()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=10, batch_size=450)

end_time = time.time()-start_time

# 4. evaluate , predict

acc = model.evaluate(x_test, y_test)
print('acc : ', acc)
print('걸린시간 : ', end_time)
# [3.3521790504455566, 0.8132399916648865]

# acc :  [4.663558483123779, 0.8110799789428711]
# 걸린시간 :  4834.566399097443

# acc :  [5.228521347045898, 0.809440016746521]
# 걸린시간 :  4679.518686771393
