import pandas as pd 
import numpy as np
import time
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
# from sklearn.linear_model import LinearRegression
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
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

print(train_set)

print('====================================================')

# def get_month(data):
#     month = data[3:5]
#     month = int(month)
#     return month

# train_set['Month'] = train_set['Date'].apply(get_month)
# test_set['Month'] = test_set['Date'].apply(get_month)

# print(train_set)
# print(test_set)

print('====================================================')
print('====================================================')

# def holiday_to_number(IsHoliday):
#     if IsHoliday == True:
#         number = 1
#     else:
#         number = 0
#         return number

train_set['IsHoliday'] = train_set['IsHoliday'].fillna(train_set.IsHoliday.dropna().mode()[0])
train_set['IsHoliday'] = train_set['IsHoliday'].apply(np.round).astype('float64')
test_set['IsHoliday'] = test_set['IsHoliday'].fillna(train_set.IsHoliday.dropna().mode()[0])
test_set['IsHoliday'] = test_set['IsHoliday'].apply(np.round).astype('float64')
print(train_set)
print(test_set)

# Weekly_Sales  Month

print('====================================================')
print('====================================================')
print('====================================================')

# model = LinearRegression()

print('====================================================')
print('====================================================')
print('====================================================')

train_set = train_set.drop(columns=['Date'])
test_set = test_set.drop(columns=['Date'])

train_set = train_set.drop(columns=['IsHoliday'])
test_set = test_set.drop(columns=['IsHoliday'])

x = train_set.drop(columns=['Weekly_Sales'])
y = train_set['Weekly_Sales']

print(train_set.head(5))

print('====================================================')
print('====================================================')

# print(train_set.shape)
# print(test_set.shape)

print('====================================================')
print('====================================================')
print('====================================================')
print('====================================================')

print(train_set.describe())

train_set['Promotion1'] = train_set['Promotion1'].fillna(train_set['Promotion1'].median())
test_set['Promotion1'] = test_set['Promotion1'].fillna(test_set['Promotion1'].median())

train_set['Promotion2'] = train_set['Promotion2'].fillna(train_set['Promotion2'].median())
test_set['Promotion2'] = test_set['Promotion2'].fillna(test_set['Promotion2'].median())

train_set['Promotion3'] = train_set['Promotion3'].fillna(train_set['Promotion3'].median())
test_set['Promotion3'] = test_set['Promotion3'].fillna(test_set['Promotion3'].median())

train_set['Promotion4'] = train_set['Promotion4'].fillna(train_set['Promotion4'].median())
test_set['Promotion4'] = test_set['Promotion4'].fillna(test_set['Promotion4'].median())

train_set['Promotion5'] = train_set['Promotion5'].fillna(train_set['Promotion5'].median())
test_set['Promotion5'] = test_set['Promotion5'].fillna(test_set['Promotion5'].median())


print('====================================================')
print('====================================================')
print('====================================================')
print('====================================================')

cat_col = train_set.dtypes[train_set.dtypes == 'object'].index   
for col in cat_col:
    train_set[col] = train_set[col].fillna('None')
    train_set[col] = LabelEncoder().fit_transform(train_set[col].values)
print(train_set.head(5))     # ==> 2줄까지 출력하여 확인  
  
cat_col = test_set.dtypes[test_set.dtypes == 'object'].index   
for col in cat_col:
    test_set[col] = test_set[col].fillna('None')
    test_set[col] = LabelEncoder().fit_transform(test_set[col].values)

y = test_set.drop(['Weekly_Sales'], axis=1)


print(train_set.isnull().sum())
# train_set = train_set.fillna(train_set.mean())
# print(train_set.shape)
# test_set = test_set.fillna(test_set.median())
print(test_set.isnull().sum())
# print(test_set.shape)

print(x)

print('====================================================')

print(train_set.head(5))


print(test_set.head(5))

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=30)

print(train_set.shape , test_set.shape)


# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))   # 0.0
print(np.max(x_train))   # 0.0 컬럼별로 나누어주어야 한다
print(np.min(x_test))
print(np.max(x_test))


# 2. model 

model = Sequential()
model.add(Dense(150, input_dim=10))
model.add(Dense(75, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear')) # softmax

# 3. compile , epochs

start_time = time.time()

earlystopping = EarlyStopping(monitor='val_loss', patience=150,
                              mode='auto', verbose=1)

model.compile(loss = 'mse', optimizer = 'adam') # categorical_crossentropy

model.fit(x_train, y_train, epochs=3, batch_size=10,
        validation_split=0.2,
        callbacks = [earlystopping],
        verbose=1)

end_time = time.time()-start_time

results = model.evaluate(x_test, y_test)


# 4. evaluate , predict

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

print('=========================================================')
def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

print('걸린시간 : ', end_time)



