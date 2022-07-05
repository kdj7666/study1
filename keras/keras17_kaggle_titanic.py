# [실습]
import pandas as pd 
from json import encoder
# ---------------------------------
from cProfile import label
import numpy as np 
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
# ---------------------------------------------------
import tensorflow as tf

# datasets.descibe()
# datasets.info()
# datasets.insull().sum()

# pandas 의 y라벨의 종류가 무엇인지 확인하는 함수 쓸것
# numpy에서는 np.unipue(y, return_counts=True)


#1. data  # 10번 경로  +는 문자가 연결이 된다
path = './_data4/titanic/' 
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)    
print(train_set)             
print(train_set.shape)   # 891 11

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)

submission = pd.read_csv(path + 'gender_submission.csv')

print(test_set)
print(test_set.shape)  # 418,10

print(train_set.columns) 

# Index(['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
#        'Fare', 'Cabin', 'Embarked'],
#       dtype='object')
print('####################################################################################')

print(test_set.columns) 
# Index(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
#        'Cabin', 'Embarked'],
#       dtype='object')

print(train_set.info())       # 컬럼에 대한 내용이 디테일하게 나온다       ( Non-Null Count ) 이빨이 빠졋다 데이터가 빠졋다  [ 결측치 ] 데이터 전처리에 아주 중요 / [이상치]라는 데이터도 있다 나중에 
print(train_set.describe())               #  describe 묘사하다 서술하다  # 최솟값 최댓값 등 확인       pd 좀더 찾아보기 중요

#### 결측치 처리 1. 제거####
print(train_set.isnull().sum())
train_set = train_set.dropna()
print(train_set.isnull().sum())
print(train_set.shape) # 183 , 11
#################################################################

x = train_set   # drop 날리다 카운트라는 줄을 날릴것이다 소숫점이 1개 
print(x)
print(x.columns)
print(x.shape)   

y = train_set  # 이렇게 하면 빠진다 지금은 이정도 ( [ ] 대괄호를 잘못치면 다 틀린다 ) 나중에 반복
print(y)
print(y.shape)   #                  벡터가 1개 그래서 최종 아웃풋 갯수는 1개   ( 여기까지가 데이터 )

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.989,
        shuffle=True,
        random_state=40)

# np.logical_or(x, y)
# print(x = train_set.info(x))
# print(train_set.dropna( subset['Age']))
# print(pd.isna('nan'))
print(x.shape)
print(y.shape)


#2. 모델구성
model = Sequential()
model.add(Dense(90, input_dim=11))          # 행 무시 열 우선 필수 
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))  
model.add(Dense(50, activation='relu'))
model.add(Dense(50))  
model.add(Dense(50))   
model.add(Dense(2, activation='sigmoid'))

#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer = 'adam',metrics=['accuracy', 'mse'])        # 평가지표는 프레딕트 결과값 어쩌구 저쩌구 해서 mse 로 가능 비슷하면 된다 

model.fit(x_train, y_train, epochs=10, batch_size=10)
from tensorflow.python.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, # mode='min'뿐아니라 max도 있음  디폴드값 찾아볼것 모르면 오토 
              restore_best_weights=True)  # < - 검색해서 정리할것 (파라미터를 적용을 시켯다 내가 하고싶은데로)
             # 모니터로 보겟다 vla_loss / patience 참다 10번 / mode = 'min'  최솟값을 verbose=1
             # 깃허브 참조 
             # 이름을 짓는다 earlystopping 변수는 첫번째를 소문자로 
             
             
#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print(a.history['val_loss'])

y_predict = model.predict(x_test)
y_predict = y_predict.flatten()
y_predict = np.where(y_predict > 0.5, 1 , 0)
print(y_predict)


from sklearn.metrics import r2_score, accuracy_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
print('acc.score : ', acc)
# print(y_predict)


y_summit = model.predict(test_set)

# y_summit = model.predict(test_set)


# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print('r2scoer :', r2)

# print(y_summit)
# print(y_summit.shape) # (715,1)

######################## .to_csv()를 사용해서 아이디값 안됨 카운트값 순서대로 
### submission.csv를 완성하시오 !!! ( 과제 겸 실습 )

dataframe = pd.DataFrame(y_summit)
dataframe.to_csv('.csv')

submission['count'] = y_summit
submission = submission.fillna(submission.mean())

# dataframe = pd.DataFrame(y_summit)
# dataframe.to_csv('.csv')

submission['count'] = y_summit        
# submission = submission.fillna(submission.mean())

submission.to_csv(path + 'submission.csv', index=False)

# earlystopping 적용 후 

# loss :  38.670223236083984
# RMSE :  56.808521319454506
# r2scoer : 0.48441298176352554

# ctivation 적용 후 

# loss :  26.252769470214844
# RMSE :  38.36484871933018
# r2scoer : 0.7648516239808246

