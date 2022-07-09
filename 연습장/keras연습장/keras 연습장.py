from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical 

text = ' 나랑 점심에 롯데리아 갈래 ? 점심 메뉴는 새우버거인데 어때 ? '

t = Tokenizer()
t.fit_on_texts([text])
print(t.word_index)

sub_text = '점심에 롯데리아 갈래? 메뉴는 새우버거인데 어때 ?'
encoded= t.texts_to_sequences([sub_text])[0]
print(encoded)


# ---------------------------------------------------

# text = x,y

# t = Tokenizer()
# t.fit_on_texts([text])
# print(t.word_index)

# encoded= t.texts_to_sequences([sub_text])[0]
# print(encoded)

# one_hot = to_categorical(encoded)
# print(one_hot)
