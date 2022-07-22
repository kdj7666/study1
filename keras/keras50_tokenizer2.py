from keras.preprocessing.text import Tokenizer

text1 = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
text2 = '나는 지구용사 이재근이다. 멋있다. 또 또 얘기해봐'

token = Tokenizer()         # 토큰아이저에 정리한것을 텍스트로 
token.fit_on_texts([text1, text2]) # 리스트형태로 받음 여러개도 가능 # 어절에 대한 토큰으로 잘라준다 크기 - 어절순서 핏 할때 인덱스가 생성 됨

print(token.word_index) # 찍는법 크기 - 어절 순서대로 인덱스가 잡힌다 
# {'마구': 1, '나는': 2, '매우': 3, '또': 4, '진짜': 5, '맛있는': 6, '밥을': 7,
# '엄청': 8, '먹었다': 9, '지구용사': 10, '이재근이다': 11, '멋있다': 12, '얘기해봐': 13}

x = token.texts_to_sequences([text1, text2]) # 배열을 수치화 시켜 숫자로 나타내어준다 
print(x)

# [[2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9], [2, 10, 11, 12, 4, 4, 13]]

from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

x_new = x[0] + x[1]
print(x_new)

# [2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9, 2, 10, 11, 12, 4, 4, 13]

x = to_categorical(x_new)
print(x)
print(x.shape) # (18, 14)

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


# ohe = OneHotEncoder(sparse=False)
# y = enco.fit_transform(y.reshape(-1,1))
# x = ohe.fit_transform(x.reshape)
# print(x)




