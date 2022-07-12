import pandas as pd 
import numpy as np
import time
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.linear_model import LinearRegression
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
# 1. data 

path = './_data/shopping/'
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)

sample_submission = pd.read_csv(path + 'sample_submission.csv',
                        index_col=0)

print(train_set.shape, test_set.shape) # (6255, 12) (180, 11)

print(train_set.columns)
# Index(['Store', 'Date', 'Temperature', 'Fuel_Price', 'Promotion1',
#        'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Unemployment',
#        'IsHoliday', 'Weekly_Sales'],
#       dtype='object')

print(test_set.columns)
# Index(['Store', 'Date', 'Temperature', 'Fuel_Price', 'Promotion1',
#        'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Unemployment',
#        'IsHoliday'],
#       dtype='object')

print('====================================================')

print(train_set.head()) # 데이터 최상단 5줄

print('====================================================')

print(train_set.tail()) # 데이터 최하단 5줄

print('====================================================')

print(train_set.info()) # 데이터 결측치 및 변수들의 타입 확인 

print('====================================================')

train_set = train_set.fillna(0)

print(train_set)

print('====================================================')

def get_month(data):
    month = data[3:5]
    month = int(month)
    return month

train_set['Month'] = train_set['Date'].apply(get_month)

print(train_set)

print('====================================================')

def holiday_to_number(isholiday):
    if isholiday == True:
        number = 1
    else:
        number = 0
        return number

train_set['NumberHoliday'] = train_set['IsHoliday'].apply(holiday_to_number)

print(train_set)

print('====================================================')


train_set = train_set.drop(columns=['Date','id', 'IsHoliday'], axis=0)
test_set = test_set.drop(columns=['Date','id' , 'IsHoliday'], axis=0)

x = train_set.drop(columns=['Weekly_Sales'], axis=1)
y = train_set['Weekly_Sales']

print(train_set.head(5))

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=30)


# 2. model 

model = Sequential()
model.add(Dense(80, input_dim=14))
model.add(Dense(200, activation='relu'))
model.add(Dense(190, activation='relu'))
model.add(Dense(180, activation='relu'))
model.add(Dense(170, activation='relu'))
model.add(Dense(160, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(75, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='relu'))

# 3. compile , epochs

start_time = time.time()
earlystopping = EarlyStopping(monitor='val_loss', patience=150,
                              mode='auto', verbose=1,
                              restore_best_weigths=True)

model.compile(loss = 'mse', optimizer = 'adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=350, batch_size=150,
          validation_split=0.2,
          callbacks = [earlystopping],
          verbose = 1)
end_time = time.time()-start_time

# 4. evaluate , predict

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print('==========================')
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('걸린시간 : ', end_time)
print('r2_score : ', r2)
