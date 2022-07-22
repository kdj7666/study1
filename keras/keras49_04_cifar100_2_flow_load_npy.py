# 넘파이에서 불러와서 모델구성
# 성능 비교 # 증폭해서 npy지점 
# 48 - 2


# 3.9.7 데이터셋은 인식 python 인식 안됌 
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import time
from keras.datasets import cifar100

# x_train, x_test, y_train, y_test = train_test_split(x,y,
#             train_size=0.7, shuffle=True, random_state=55)
# np.save('d:/study_data/_save/_npy/keras49_4_train_x(cifar100).npy', arr=x_train)
# np.save('d:/study_data/_save/_npy/keras49_4_train_y(cifar100).npy', arr=y_train)
# np.save('d:/study_data/_save/_npy/keras49_4_test_x(cifar100).npy', arr=x_test)
# np.save('d:/study_data/_save/_npy/keras49_4_test_y(cifar100).npy', arr=y_test)
# # 넌파이 파일로 저장한다 넌파일수치로 저장이 됨

x_train = np.load('d:/study_data/_save/_npy/keras49_4_train_x(cifar100).npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_4_train_y(cifar100).npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_4_test_x(cifar100).npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_4_test_y(cifar100).npy')

print(x_train.shape) # (50400, 32, 32, 3)
print(y_train.shape) # (50400,)
print(x_test.shape) # (10000, 32, 32, 3)
print(y_test.shape) # (10000,)

#2. model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(32, 32, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. compile, epochs

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=10, batch_size=32,
          validation_split=0.2, verbose=1) # 허나 배치를 최대로 잡으면 이것도 가능하다


# hist = model.fit_generator(x_train,y_train epochs=10, steps_per_epoch=32,    
#                          # 스텝 펄 에포 ( 통상적으로 batch= 160/5 = 32)  # 훈련 배치 사이즈가 32가 넘어서도 돌아가긴한다 추가적 환경 제공 가능
#                     validation_data=xy_test,                    # 발리데이션 범주를 테스트로
#                     validation_steps=2, verbose=1)                         # 발리데이션 스텝 : 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다

end_time = time.time()-start_time

#4. evluate, predict
acc = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1]) # 마지막 괄호로 마지막 1개만 보겠다
print('val_loss : ', val_loss[-1])
print('val_accuracy : ', val_accuracy[-1])
print('accuracy : ', acc[-1])
print('걸린시간 : ', end_time)


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



# loss :  3004.281494140625
# val_loss :  2996.620849609375
# val_accuracy :  0.009920635260641575
# accuracy :  0.010074956342577934
# 걸린시간 :  18.302319765090942

# loss :  752.1240844726562
# val_loss :  835.9088745117188
# val_accuracy :  0.010333333164453506
# accuracy :  0.00936111155897379
# 걸린시간 :  123.87287139892578