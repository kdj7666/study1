
# kaggle house 문제풀이 
from operator import index
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import nan_euclidean_distances, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame

#1. data  # 10번 경로  +는 문자가 연결이 된다

path = './_data3/house/' 
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0) 
print(train_set)             
print(train_set.shape)  # 1460 , 80

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0) 

submission = pd.read_csv(path + 'sample_submission.csv')

print(test_set)
print(test_set.shape)  # 1459 , 79

#print(train_set.columns) 
print(train_set.info())    
print(train_set.describe()) 

# train , test 값 문자열을 날리고 변환시켜야함 
x = train_set.drop(['SalePrice'], axis=1)  
print(x)
#print(x.columns)
print(x.shape)  # 1460 , 79

y = train_set['SalePrice']  
print(y)
print(y.shape)  # 1460 ,

label_obj_list = index(['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',    
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',     
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',   
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',     
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',       
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',      
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',   
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',   
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',  
       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',   
       'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice'],
      dtype='object')
                      

laber_obj_list = index(['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',    
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',     
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',   
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',     
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',       
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',      
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',   
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',   
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',  
       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',   
       'MoSold', 'YrSold', 'SaleType', 'SaleCondition'],
      dtype='object')

encoder = LabelEncoder()

for obj in label_obj_list:
    encoder = LabelEncoder()
    encoder.fit(list(DataFrame[obj].values))
    DataFrame[obj] = encoder.transform(list(DataFrame[obj].values))


##################################


# #### 결측치 처리 1. 제거####
# print(train_set.isnull().sum())
# train_set = train_set.dropna()
# print(train_set.isnull().sum())
# print(train_set.shape)

null = (DataFrame.isna().sum() / len(DataFrame) * 100)

null = null.drop(null[null == 0].index).sort_values(ascending=False)
null

'''
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7,
        shuffle=True,
        random_state=50)

# np.logical_or(x, y)
# print(x = train_set.info(x))
# print(train_set.dropna( subset['Age']))
# print(pd.isna('nan'))

#2. 모델구성
model = Sequential()
model.add(Dense(90, input_dim=79))          # 행 무시 열 우선 필수 
model.add(Dense(80, activation='swish'))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer = 'adam')        # 평가지표는 프레딕트 결과값 어쩌구 저쩌구 해서 mse 로 가능 비슷하면 된다 
model.fit(x_train, y_train, epochs=100, batch_size=100) 

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_summit = model.predict(test_set)

print(y_summit)
print(y_summit.shape)

# submission['count'] = y_summit        
# submission = submission.fillna(submission.mean())
# submission.to_csv(path + 'submission.csv', index=False)


'''