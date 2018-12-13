"""-*- coding: utf-8 -*-
 DateTime   : 2018/12/13 17:35
 Author  : Peter_Bonnie
 FileName    : VGG.py
 Software: PyCharm
"""
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Softmax,Activation,Dense
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from sklearn.metrics import recall_score,f1_score,precision_score
import numpy as np

#运用VGG进行图像分类与目标定位


#加载数据集
data=mnist.load_data()
(X_train,Y_train),(X_test,Y_test)=data
X_train=X_train.reshape(-1,28,28,1)  #这里我们使用的是黑白图片
X_test=X_test.reshape(-1,28,28,1)
Y_train=to_categorical(Y_train,num_classes=10)
Y_test=to_categorical(Y_test,num_classes=10)

#模型搭建
def VGG(X,Y):
    model=Sequential()
    #layer_1
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=X.shape[1:],padding='same',data_format='channels_last',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',data_format='channels_last',kernel_initializer='uniform',activation='relu'))
    model.add(MaxPooling2D((2,2)))
    #layer_2
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(128,(2,2),strides=(1,1),padding='same',data_format='channels_last',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D((2,2)))
    #layer_3
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(256, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    #layer_4
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(Conv2D(512, (1,1), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    #layer_5
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(Conv2D(512, (1,1), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    #全连接层+
    model.add(Flatten())  #拉平
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(1000,activation='relu'))
    model.add(Dense(10,activation='softmax'))

    model.summary()
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model

if __name__=="__main__":
    model=VGG(X_train,Y_train)
    model.fit(X_train,Y_train,batch_size=128,epochs=1)
    # Y_predict=model.predict(X_train)
    # Y_predict=np.argmax(Y_predict,axis=1)
    # print("recall_score:",recall_score(Y_train, Y_predict))
    # print("f1_score:",f1_score(Y_train, Y_predict))
    # print("precision_score:",precision_score(Y_train, Y_predict))
    loss,acc=model.evaluate(X_test,Y_test)
    print(loss,acc)