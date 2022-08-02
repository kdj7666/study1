import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

# 1. data

# np.save('d:/study_data/_save/_npy/keras47_4_train_x(men_women).npy', arr=x_train)
# np.save('d:/study_data/_save/_npy/keras47_4_train_y(men_women).npy', arr=y_train)
# np.save('d:/study_data/_save/_npy/keras47_4_test_x(men_women).npy', arr=x_test)
# np.save('d:/study_data/_save/_npy/keras47_4_test_y(men_women).npy', arr=y_test)
# np.save('d:/study_data/_save/_npy/keras47_4_test_k(men_women).npy', arr=kim[0][0])
# # 넌파이 파일로 저장한다 넌파일수치로 저장이 됨

x_train = np.load('d:/study_data/_save/_npy/keras53-13_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras53-13_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras53-13_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras53-13_test_y.npy')
k_test = np.load('d:/study_data/_save/_npy/keras53-13_test_k.npy')
print(x_train.shape) # (1700, 150, 150, 3)
print(y_train.shape) # (1700, 5)
print(x_test.shape) # (300, 150, 150, 3)
print(y_test.shape) # (300, 5 )
print(k_test.shape) # (1, 150 , 150 , 3)


#2. model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, MaxPool2D

model = Sequential()
model.add(Conv2D(128,(2,2),input_shape=(150,150,3),padding='same',activation='relu'))
# model.add(conv_base)
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
model.add(MaxPool2D((2,2)))
# model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
# model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.6))                 #과적합방지
model.add(Dense(5,activation='softmax'))
model.summary()

#3. compile, epochs

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es =EarlyStopping(monitor='loss', patience=35, mode='min', 
              verbose=1, restore_best_weights = True)

start_time = time.time()

hist = model.fit(x_train,y_train, epochs=45, batch_size=35,
          validation_split=0.2, verbose=1,
        #   callbacks=[es],
          ) # 허나 배치를 최대로 잡으면 이것도 가능하다


# hist = model.fit_generator(xy_train, epochs=10, steps_per_epoch=32,    
#                          # 스텝 펄 에포 ( 통상적으로 batch= 160/5 = 32)  # 훈련 배치 사이즈가 32가 넘어서도 돌아가긴한다 추가적 환경 제공 가능
#                     validation_data=xy_test,                    # 발리데이션 범주를 테스트로
#                     validation_steps=2, verbose=1)                         # 발리데이션 스텝 : 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다

end_time = time.time()-start_time


acc = hist.history['accuracy']
# val_accuracy = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# print('loss : ', loss[-1]) # 마지막 괄호로 마지막 1개만 보겠다
# print('val_loss : ', val_loss[-1])
# print('val_accuracy : ', val_accuracy[-1])
print('accuracy : ', acc[-1])
print('걸린시간 : ', end_time)

#4. evaluate, predict

# loss = model.evaluate(x_test, y_test)
# print("loss :",loss[-1])
# print("==========================================")


y_test2 = [0,1,2,3]
y_predict = model.predict(k_test)
y_test = np.argmax(y_test, axis= 1)
y_predict = np.argmax(y_predict, axis=1)
print('predict : ',y_predict)
print('2')
acc = accuracy_score(y_test2,y_predict)
print('acc : ',acc)

if y_predict[0] == 0:
    print('검지')
elif  y_predict[0] ==1 :
    print('소지')
elif  y_predict[0] ==2 :
    print('약지')
elif  y_predict[0] ==3 :
    print('엄지')
elif  y_predict[0] ==4 :
    print('중지')        



# accuracy :  0.6264705657958984
# 걸린시간 :  92.64283895492554
# ==
# predict :  [4]
# ==
# 22222222
# acc :  1.0
# 중지

# accuracy :  0.625
# 걸린시간 :  92.93881940841675
# ==
# predict :  [0]
# ==
# 22222222
# acc :  0.0
# 검지

# accuracy :  0.6205882430076599
# 걸린시간 :  92.5685567855835
# predict :  [3]
# 2
# acc :  0.0
# 엄지

