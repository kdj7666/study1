from subprocess import call
import pandas as pd

#describe info insullsum 
#y라벨의 종류가 무엇인지 확인하는 판다스 함수 =  np.unique
path = './_data4/titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

print(train_set.shape)
print(test_set.shape)
print(test_set.columns)
print(train_set.columns)



# pd.Series.unique(train_set)
# pd.Series.value_counts(train_set)
# (891, 12)
# (418, 11)
# Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
#        'Ticket', 'Fare', 'Cabin', 'Embarked'],
#       dtype='object')
# Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
#       dtype='object')
print(train_set.info())
print(test_set.info())
print(train_set.describe())
print(test_set.describe())
 #   Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   PassengerId  891 non-null    int64
#  1   Survived     891 non-null    int64
#  2   Pclass       891 non-null    int64
#  3   Name         891 non-null    object
#  4   Sex          891 non-null    object
#  5   Age          714 non-null    float64
#  6   SibSp        891 non-null    int64
#  7   Parch        891 non-null    int64
#  8   Ticket       891 non-null    object
#  9   Fare         891 non-null    float64
#  10  Cabin        204 non-null    object
#  11  Embarked     889 non-null    object
 
 
#  ---  ------       --------------  -----
#  0   PassengerId  418 non-null    int64
#  1   Pclass       418 non-null    int64
#  2   Name         418 non-null    object
#  3   Sex          418 non-null    object
#  4   Age          332 non-null    float64
#  5   SibSp        418 non-null    int64
#  6   Parch        418 non-null    int64
#  7   Ticket       418 non-null    object
#  8   Fare         417 non-null    float64
#  9   Cabin        91 non-null     object
#  10  Embarked     418 non-null    object

print(train_set.isnull().sum())
print(test_set.isnull().sum())
print(train_set['Survived'].unique())
print(train_set['Survived'].value_counts())

train_set=train_set.dropna()
test_set=test_set.dropna()

#일단 떨군다


#        PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
# count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
# mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
# std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
# min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
# 25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
# 50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
# 75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
# max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
#        PassengerId      Pclass         Age       SibSp       Parch        Fare
# count   418.000000  418.000000  332.000000  418.000000  418.000000  417.000000
# mean   1100.500000    2.265550   30.272590    0.447368    0.392344   35.627188
# std     120.810458    0.841838   14.181209    0.896760    0.981429   55.907576
# min     892.000000    1.000000    0.170000    0.000000    0.000000    0.000000
# 25%     996.250000    1.000000   21.000000    0.000000    0.000000    7.895800
# 50%    1100.500000    3.000000   27.000000    0.000000    0.000000   14.454200
# 75%    1204.750000    3.000000   39.000000    1.000000    0.000000   31.500000
# max    1309.000000    3.000000   76.000000    8.000000    9.000000  512.329200


#패신저 아이디 삭제

#xy 설정하고 테스트 트레인셋 나눈고 모델구성 


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

x = train_set.drop(['Survived','Name', 'Sex','Ticket', 'Cabin','Embarked'], axis=1)
y = train_set['Survived']
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=137)

model = Sequential()
model.add(Dense(5, input_dim=5))
# print(x.shape) #5
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#컴파일 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs = 10, batch_size=32, validation_split=0.2, callbacks=ES)

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print(y_test)
print(y_predict)
print(y_test.shape)

import numpy as np
y_predict = np.argmax(y_predict, axis= 1)

#패신저 아이디가 왜 y_test에 붙어서 나오는가?


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)

print('loss', loss)
print('acc', acc)

#현재 문제 y_predict값 nan값 나옴 이해못함 아마도 라벨인코딩 argmax 조합인것 같은데 다시 봐야함
#pandas 함수 적용못함

# y_summit = model.predict(test_set)
# submission = pd.read_csv(path + 'gender_submission.csv')
# submission['Survive'] = y_summit
# submission.to_csv(path + 'submission.csv')
# ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).

#현재 라벨인코딩 밋 원핫인코딩 3개 쓸수 없음으로 인해서 데이터셋 na를 전부 날렸다 알게되면 원핫인코딩으로 숫자변환해 쓸수 있을 것 추후 전처리를 배우면 상관관계를 분석해 이용할 수?
#argmax 의 의의를 제대로 이해못함
#아침에 와서 모든 기술이 다 들어간 코드 작성해보기
