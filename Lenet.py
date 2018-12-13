"""-*- coding: utf-8 -*-
 DateTime   : 2018/12/13 10:22
 Author  : Peter_Bonnie
 FileName    : Lenet.py
 Software: PyCharm
"""

"""导入库"""
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras.utils import plot_model
# import pydot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import KFold
# import os
# os.environ["PATH"] += os.pathsep + 'D:\Anaconda\\release\\bin'
# print(os.environ["PATH"])

"""准备数据集"""
dataset=mnist.load_data()
(X_train,Y_train),(X_test,Y_test)=dataset
X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)
Y_train=to_categorical(Y_train,num_classes=10)
Y_test=to_categorical(Y_test,num_classes=10)

"""搭建模型"""
def LeNet(X_train,Y_train):
    model=Sequential()
    model.add(Conv2D(filters=5,kernel_size=(3,3),strides=(1,1),input_shape=X_train.shape[1:],padding='same',
                     data_format='channels_last',activation='relu',kernel_initializer='uniform'))  #[None,28,28,5]
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2,2)))  #池化核大小[None,14,14,5]

    model.add(Conv2D(16,(3,3),strides=(1,1),data_format='channels_last',padding='same',activation='relu',kernel_initializer='uniform'))#[None,12,12,16]
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(2,2))  #output_shape=[None,6,6,16]

    model.add(Conv2D(32, (3, 3), strides=(1, 1), data_format='channels_last', padding='same', activation='relu',
                     kernel_initializer='uniform'))   #[None,4,4,32]
    model.add(Dropout(0.2))
    # model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(100,(3,3),strides=(1,1),data_format='channels_last',activation='relu',kernel_initializer='uniform'))  #[None,2,2,100]
    model.add(Flatten(data_format='channels_last'))  #[None,400]
    model.add(Dense(168,activation='relu'))   #[None,168]
    model.add(Dense(84,activation='relu'))    #[None,84]
    model.add(Dense(10,activation='softmax'))  #[None,10]
    #打印参数
    model.summary()
    #编译模型
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

if __name__=="__main__":
    #模型训练
    model=LeNet(X_train,Y_train)
    model.fit(x=X_train,y=Y_train,batch_size=128,epochs=1)
    #模型评估
    loss,acc=model.evaluate(x=X_test,y=Y_test)
    print("loss:{}===acc:{}".format(loss,acc))

    plot_model(model=model,to_file='lenet.png',show_shapes=True)
    le=mpimg.imread('lenet.png')
    plt.imshow(le)
    plt.axis('off')
    plt.show()
# C:\Program Files (x86)\Graphviz2.38\bin