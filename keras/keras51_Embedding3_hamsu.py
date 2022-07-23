# keras51 - 1 
from keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, LSTM, Conv1D, Embedding, Input, MaxPooling2D, MaxPooling1D
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
# {'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6,
# '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10,
# '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글세요': 16,
# '별로에요': 17, '생각보다': 18, '지루해요': 19,
# '인기가': 20, '어색해요': 21, '재미없어요': 22,
# '재미없다': 23, '재밌네요': 24, '인수가': 25, '못': 26,
# '생기긴': 27, '했어요': 28, '안결': 29, '혼해요': 30}


x = token.texts_to_sequences(docs) # 문장별로 토큰 수치화 [] 없는건 알아서 
print(x)           # 수치화가 1개인 경우와 5개인경우 아구가 안맞다 그러니 제일 큰 아구에 작은애들을 맞춰준다
# 0을 채워주어 다 같은 수치로 변경 너무 큰 수치같은 경우엔 잘라 버린다  / 0을 통상 앞에서 채운다 하지만 정답은 아니다 뒤에도 채워볼것
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17],
#  [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27, 28], [29, 30]]

# 0 채우기 위해 부른것 

pad_x = pad_sequences(x,padding='pre', maxlen=5, truncating='pre') # maxlen 최대 글자 수치 = 5 / 5글자까지 하겠다 pre 앞  post 뒤 

print(pad_x)
print(pad_x.shape) # 14 , 5  ( Dense )       ( LSTM conv1d )  연속적인 데이터이기에 reshape로 차원을 늘려도 된다

word_size = len(token.word_index) # len 갯수 
print('word_size :', word_size) # 단어사전의 갯수는 30 개   워드 사이즈가 있다  나중에 나오니 찾아볼것

print(np.unique(pad_x, return_counts=True))

# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])


#2. model 

model = Sequential()      #  14 , 5 ( 원핫이 안되어있는 수치의 shape )

input1 = Input(shape=(5,))     # 행 무시 열 우선 5 , 
dense1 = Embedding(input_dim=31, output_dim=10, input_length=5)(input1)
dense2 = LSTM(32)(dense1)
output1 = Dense(1, activation='sigmoid')(dense2)
model = Model(inputs=input1, outputs=output1)
model.summary()


                     # 단어사전의 갯수 # 아웃풋 노드의 갯수는 본인이 정하는 것                            
# model.add(Embedding(input_dim=31, output_dim=10, input_length=5)) # 이렇게 나온 수치는 통상 3차원으로 나옴
# model.add(Embedding(input_dim=31, output_dim=10) # input_length=5 길이의 갯수를 정해준다 없으면 자동으로 정해준다 ( 명확히 알고있으면 정해주는게 좋다 ) # 숫자만 넣어주면 에러가 난다 자주쓰이지 않기에 
# model.add(Embedding(31, 10, input_length=5))
# model.add(Embedding(31, 10))
# model.add(LSTM(32))                # output dim 은 y의 갯수 input length 는 다시 
# model.add(Dense(1, activation='sigmoid')) # 참 거짓을 밝히기에 sigmoid 


# 3. compile , epochs
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=3, batch_size=16)


# 4. evaluate , predict
acc = model.evaluate(pad_x, labels)[1]
print('acc : ', acc)






