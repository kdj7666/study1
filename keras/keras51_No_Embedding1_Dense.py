# keras51 - 1 
from keras.preprocessing.text import Tokenizer
import numpy as np
from sqlalchemy import Float
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Conv1D, Embedding
from keras.preprocessing.sequence import pad_sequences

#1. data 
docs = ['너무 재밌어요','참 최고에요','참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다.','한 번 더 보고 싶네요','글세요',
        '별로에요','생각보다 지루해요','인기가 어색해요','재미없어요',
        '너무 재미없다','참 재밌네요','인수가 못 생기긴 했어요','안결 혼해요']

#  긍정 1 부정 0 
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])  # 14  

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)



pad_x = pad_sequences(x,padding='pre', maxlen=5, truncating='pre')

print(pad_x)
print(pad_x.shape) # 14,5

word_size = len(token.word_index)
print('word_size :', word_size)

print(np.unique(pad_x, return_counts=True))

#2. model 

model = Sequential()     
# model.add(Embedding(input_dim=31, output_dim=10, input_length=5))
# model.add(LSTM(32))
model.add(Dense(32, input_shape=(5,)))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 3. compile , epochs
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=3, batch_size=16)

# 4. evaluate , predict
acc = model.evaluate(pad_x, labels)[1]
print('acc : ', acc)

# [ 실습 ] # 

# x_predict = '나는 형권이가 정말 재미없다 너무 정말 '


# #  긍정 1 부정 0 
# x_ll = np.array([x_predict])  # 14 

# token = Tokenizer()
# token.fit_on_texts(x_predict)
# print(token.word_index)

# x_ll = token.texts_to_sequences(x_predict)
# print(x)

# x_ll = pad_sequences(x_ll, padding='pre')
# y_predict = model.predict(x_ll)
# print(y_predict)
# score = Float(model.predict(x_ll))
# # 4. evaluate , predict
# acc = model.evaluate(pad_x, labels)[1]
# print('acc : ', acc)
# # 결과는 긍정? 부정? 




























