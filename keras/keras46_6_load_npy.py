# 46 - 2 번 복붙
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time

# from tensorflow.python.keras import image
# from tensorflow.python.keras import ImageDataGenerator
# 대문자는 90%이상 클래스

# 1. data

# np.save('d:/study_data/_save/_npy/keras46_5_train_x.npy', arr=xy_train[0][0])
# np.save('d:/study_data/_save/_npy/keras46_5_train_y.npy', arr=xy_train[0][1])
# np.save('d:/study_data/_save/_npy/keras46_5_test_x.npy', arr=xy_test[0][0])
# np.save('d:/study_data/_save/_npy/keras46_5_test_y.npy', arr=xy_test[0][1])
# 넌파이 파일로 저장한다 넌파일수치로 저장이 됨

x_train = np.load('d:/study_data/_save/_npy/keras46_5_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras46_5_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras46_5_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras46_5_test_y.npy')

print(x_train.shape) # (160, 150, 150, 1)
print(y_train.shape) # (160,)
print(x_test.shape) # (120, 150, 150, 1)
print(y_test.shape) # (120,)









'''
#2. model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(200, 200, 1), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. compile, epochs

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(xy_train[0][0], xy_train[0][1]) # 허나 배치를 최대로 잡으면 이것도 가능하다
start_time = time.time()
hist = model.fit_generator(xy_train, epochs=150, steps_per_epoch=64,    
                         # 스텝 펄 에포 ( 통상적으로 batch= 160/5 = 32)  # 훈련 배치 사이즈가 32가 넘어서도 돌아가긴한다 추가적 환경 제공 가능
                    validation_data=xy_test,                    # 발리데이션 범주를 테스트로
                    validation_steps=5)                         # 발리데이션 스텝 : 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다
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

# loss :  0.693220853805542
# val_loss :  0.6934398412704468
# val_accuracy :  0.3499999940395355
# accuracy :  0.5


# 그림그리기 

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss & val_loss')    
plt.title('로스값과 검증로스값')    
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')   # 우측상단에 라벨표시
plt.legend()   # 자동으로 빈 공간에 라벨표시
plt.show()

'''
