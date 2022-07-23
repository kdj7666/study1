from keras.preprocessing.text import Tokenizer
import numpy as np
test = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()         # 토큰아이저에 정리한것을 텍스트로 
token.fit_on_texts([test]) # 리스트형태로 받음 여러개도 가능 # 어절에 대한 토큰으로 잘라준다 크기 - 어절순서 핏 할때 인덱스가 생성 됨

print(token.word_index) # 찍는법 크기 - 어절 순서대로 인덱스가 잡힌다 {'진짜': 1, '마구': 2, '매우': 3, '나는': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}

x = token.texts_to_sequences([test]) # 배열을 수치화 시켜 숫자로 나타내어준다 
print(x)

from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

x = to_categorical(x)
print(x)
print(x.shape)

# [[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
# (1, 11, 9) 3차원 LSTM 을 생각해야한다 


ohe = OneHotEncoder(sparse=False)
x_new = np.array(x)

xb = ohe.fit_transform(x_new.reshape(-1,1))
print(xb)
print(xb.shape)



