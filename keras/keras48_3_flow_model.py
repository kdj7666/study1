# 3.9.7 데이터셋은 인식 python 인식 안됌 
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import time

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

augment_size = 40000        # 증강 사이즈 
randidx = np.random.randint(x_train.shape[0], size=augment_size)  # np.random.randint = 무작위로 int(정수)값을 넣어준다 
print(randidx) # [20762  8095 11489 ... 58612  1518 52314]
print(x_train.shape) # (60000, 28, 28 )
print(x_train.shape[0]) # 60000
print(np.min(randidx), np.max(randidx))  # 0 59999  랜덤 난수 조절 가능 
print(type(randidx))  # <class 'numpy.ndarray'>

# 카피 등장

x_augmented = x_train[randidx].copy()     
y_augmented = y_train[randidx].copy()      # x의 값만 뽑을수 없다 y의 값도 같은위치에 같은걸로 만들어줘야한다 

print(x_augmented.shape)   # (40000, 28, 28)  카피본 
print(y_augmented.shape)   # (40000,)

# 원본 등장 

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)

x_augmented = train_datagen.flow(x_augmented,y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False).next()[0]  # 0번째가 x 1번째가 y로

print(x_augmented)
print(x_augmented.shape) # 40000 28 28 1

# concatenate 사슬처럼 엮다 괄호 2개를 제공 필수임 나중에배우지만 찾아서 공부할것
x_train = np.concatenate((x_train, x_augmented)) 
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)  # (100000, 28, 28, 1) (100000,)


#### 모델 구성
# 성능비교, 증폭 전 후 비교 

#2. model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. compile, epochs

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=5, batch_size=15,
          validation_split=0.1, verbose=1) # 허나 배치를 최대로 잡으면 이것도 가능하다


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



# 그림그리기 

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.plot(hist.history['val_accuracy'], marker='.', c='pink', label='val_accuracy')
plt.plot(hist.history['accuracy'], marker='.', c='green', label='accuracy')
plt.grid()
plt.title('loss & val_loss')    
plt.title('로스값과 검증로스값')    
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')   # 우측상단에 라벨표시
plt.legend()   # 자동으로 빈 공간에 라벨표시
plt.show()

# loss :  0.6771479249000549
# val_loss :  0.6616235375404358
# val_accuracy :  0.6499999761581421
# accuracy :  0.5874999761581421
# 걸린시간 :  478.4866187572479



# loss :  1.477608561515808
# val_loss :  1.950744390487671
# val_accuracy :  0.10040000081062317
# accuracy :  0.12806667387485504
# 걸린시간 :  92.03242325782776