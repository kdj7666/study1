import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# 1. 데이터
x_train = np.load('d:/study_data/_save/_npy/keras49_8_train_x(horse-or-human).npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_8_train_y(horse-or-human).npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_8_test_x(horse-or-human).npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_8_test_y(horse-or-human).npy')

print(x_train.shape) # (1500, 50, 50, 1)
print(y_train.shape) # (1500,)
print(x_test.shape) # (500, 50, 50, 1)
print(y_test.shape) # (500,)


# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(50,50,1), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 메트릭스에 'acc'해도 됨
log = model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=13)

# 그래프
loss = log.history['loss']
accuracy = log.history['accuracy']
val_loss = log.history['val_loss']
val_accuracy = log.history['val_accuracy']

print('loss: ', loss[-1])
print('accuracy: ', accuracy[-1])
print('val_loss: ', val_loss[-1])
print('val_accuracy: ', val_accuracy[-1])

# import matplotlib as mpl
# import matplotlib.pyplot as plt

# mpl.rcParams['font.family'] = 'malgun gothic'
# mpl.rcParams['axes.unicode_minus'] = False

# plt.figure(figsize=(9,6))
# plt.plot(log.history['loss'], c='black', label='loss')
# plt.plot(log.history['accuracy'], marker='.', c='red', label='accuracy')
# plt.plot(log.history['val_loss'], c='blue', label='val_loss')
# plt.plot(log.history['val_accuracy'], marker='.', c='green', label='val_accuracy')
# plt.grid()
# plt.title('뇌 사진')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend()
# plt.show()

# loss:  0.789369523525238
# accuracy:  0.5774999856948853
# val_loss:  1.6903252601623535
# val_accuracy:  0.3799999952316284


# loss:  0.32662373781204224
# accuracy:  0.7691666483879089
# val_loss:  2.032010555267334
# val_accuracy:  0.6433333158493042

