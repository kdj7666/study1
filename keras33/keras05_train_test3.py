import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. date
x = np.array([1,2,3,4,5,6,7,8,9,10])  
y = np.array([1,2,3,4,5,6,7,8,9,10])
# x_train =np.array([1,2,3,4,5,6,7])
# x_test = np.array([8,9,10])
# y_train =np.array([1,2,3,4,5,6,7])
# y_test = np.array([8,9,10])

# [검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법 찾아라 
                                                                 # train_test_split 를 트레인 테스트에 해달ㄹ ㅏ
from sklearn.model_selection import train_test_split             # 데이터 훈련은 전체를 다 해주고 그중 일부를 섞어 작업을 해준다 
x_train, x_test, y_train, y_text = train_test_split(             #train , test 분리해서 작업 해야 하고 간섭이 없어야 한다 ( 필수 )
    x,y, test_size=0.3,  
    train_size=0.7,                      # 전체 데이터 70%를 훈련 시켜라 
    # shuffle=True,                      #shuffle = True = 섞다    shuffle = False = 순차적으로 
    random_state=66
) 
# 랜덤 난수값을 66 
# test_size= 0.4 , train_size= 0.7 일때 1 이상 초과값이나와 오류
# x_train, x_tset, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False) # True = 무작위 False = 순차적으로

print(x_train)
print(x_test)
print(y_train)
print(y_text)
                     #첫번째 모델에 들어간 데이터  랜덤값으로 들어간 경우 데이터가 다르면 두번째 모델에 들어간 데이터값이 다르다 
                     #두번째 모델에 들어간 데이터 첫번째 모델에서 나온 데이터가 다를경우 데이터값이 다르다 


#2. model             딥러닝 인풋1개 아웃풋1개 히든레이어 10개
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

#3. 컴파일 , 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=400 , batch_size=1)

#4. 평가 예측                                            # 훈련은 다 시켜주면 좋다 
loss = model.evaluate(x, y)
print('loss', loss)
result = model.predict([11])
print('11의 예측값 : ',result)

