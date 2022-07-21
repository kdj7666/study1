# 실습
# 본인 사진으로 predict 하시오 !!! 
# d:/study_data/_data/image/ 안에 넣고 


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
from sklearn.model_selection import train_test_split

# 1. data




# np.save('d:/study_data/_save/_npy/keras47_4_train_x(men_women).npy', arr=x_train)
# np.save('d:/study_data/_save/_npy/keras47_4_train_y(men_women).npy', arr=y_train)
# np.save('d:/study_data/_save/_npy/keras47_4_test_x(men_women).npy', arr=x_test)
# np.save('d:/study_data/_save/_npy/keras47_4_test_y(men_women).npy', arr=y_test)
# np.save('d:/study_data/_save/_npy/keras47_4_test_k(men_women).npy', arr=kim[0][0])
# # 넌파이 파일로 저장한다 넌파일수치로 저장이 됨

x_train = np.load('d:/study_data/_save/_npy/keras47_4_train_x(men_women).npy')
y_train = np.load('d:/study_data/_save/_npy/keras47_4_train_y(men_women).npy')
x_test = np.load('d:/study_data/_save/_npy/keras47_4_test_x(men_women).npy')
y_test = np.load('d:/study_data/_save/_npy/keras47_4_test_y(men_women).npy')
k_test = np.load('d:/study_data/_save/_npy/keras47_4_test_k(men_women).npy')
print(x_train.shape) # (2316, 100, 100, 3)
print(y_train.shape) # (2316,)
print(x_test.shape) # (993, 100, 100, 3)
print(y_test.shape) # (993, 100, 100, 3)
print(k_test.shape) # (  1, 100 , 100 , 3)


#2. model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(100, 100, 3),padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. compile, epochs

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()

hist = model.fit(x_train,y_train, epochs=15, batch_size=100,
          validation_split=0.2, verbose=1) # 허나 배치를 최대로 잡으면 이것도 가능하다


# hist = model.fit_generator(xy_train, epochs=10, steps_per_epoch=32,    
#                          # 스텝 펄 에포 ( 통상적으로 batch= 160/5 = 32)  # 훈련 배치 사이즈가 32가 넘어서도 돌아가긴한다 추가적 환경 제공 가능
#                     validation_data=xy_test,                    # 발리데이션 범주를 테스트로
#                     validation_steps=2, verbose=1)                         # 발리데이션 스텝 : 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다

end_time = time.time()-start_time


acc = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1]) # 마지막 괄호로 마지막 1개만 보겠다
print('val_loss : ', val_loss[-1])
print('val_accuracy : ', val_accuracy[-1])
print('accuracy : ', acc[-1])
print('걸린시간 : ', end_time)

#4. evaluate, predict

loss = model.evaluate(x_test, y_test)
print("loss :",loss)
print("====================")


y_predict = model.predict(k_test)
print(y_predict)

if 	y_predict >= 0.5 :
    print('여자다') # 출력값: 
else :
    print('남자다') # 출력값:
print('keras47_4_에ㅔ베베')

# # 그림그리기

# import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc

# font_path = 'C:\Windows\Fonts\malgun.ttf'
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.plot(hist.history['val_accuracy'], marker='.', c='pink', label='val_accuracy')
# plt.plot(hist.history['accuracy'], marker='.', c='green', label='accuracy')
# plt.grid()
# plt.title('loss & val_loss')    
# plt.title('로스값과 검증로스값')    
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(loc='upper right')   # 우측상단에 라벨표시
# plt.legend()   # 자동으로 빈 공간에 라벨표시
# plt.show()


# loss :  0.6771479249000549
# val_loss :  0.6616235375404358
# val_accuracy :  0.6499999761581421
# accuracy :  0.5874999761581421
# 걸린시간 :  478.4866187572479



# loss :  9.948715887730941e-05
# val_loss :  5.168074494577013e-05
# val_accuracy :  1.0
# accuracy :  1.0
# 걸린시간 :  929.3208270072937
