#!usr/bin/env python
#encoding:utf-8

from __future__ import division


#功能： 构建人脸识别模型


import os
import cv2
import random
import numpy as np
from PIL import Image
from keras.utils import np_utils
from keras.models import Sequential,load_model
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Dropout
#使用keras，convolution，maxpooling，flatten，dropout


class DataSet(object):
    #建立用于存储和格式化读取训练数据的类
    def __init__(self,path):
        #初始化
        self.num_classes=None
        # 训练集 测试集
        self.X_train=None
        self.X_test=None
        self.Y_train=None
        self.Y_test=None
        self.img_size=128
        #模型接受的人脸图片一定得是128*128的
        self.extract_data(path)
        #在这个类初始化的过程中读取path下的训练数据


    def extract_data(self,path):
        #抽取数据
        imgs,labels,counter=read_file(path)
        #根据指定路径读取出图片，标签和类别数
        #将数据集打乱随机分组
        X_train,X_test,y_train,y_test=train_test_split(imgs,labels,test_size=0.2,random_state=random.randint(0, 100))
        #重新格式化和标准化
        X_train=X_train.reshape(X_train.shape[0], 1, self.img_size, self.img_size)/255.0
        X_test=X_test.reshape(X_test.shape[0], 1, self.img_size, self.img_size)/255.0
        # 像素数据浮点化以便归一化
        X_train=X_train.astype('float32')
        X_test=X_test.astype('float32')
        #将labels转成binary class matrices
        Y_train=np_utils.to_categorical(y_train, num_classes=counter)
        Y_test=np_utils.to_categorical(y_test, num_classes=counter)
        #将格式化后的数据赋值给类的属性上
        self.X_train=X_train
        self.X_test=X_test
        self.Y_train=Y_train
        self.Y_test=Y_test
        self.num_classes=counter


    def check(self):
        #进行校验
        print('num of dim:', self.X_test.ndim)
        print('shape:', self.X_test.shape)
        print('size:', self.X_test.size)
        print('num of dim:', self.X_train.ndim)
        print('shape:', self.X_train.shape)
        print('size:', self.X_train.size)


def endwith(s,*endstring):
    #再次对字符串的后续和标签进行匹配
    resultArray = map(s.endswith,endstring)
    if True in resultArray:
        return True
    else:
        return False


def read_file(path):
    #图片读取
    img_list=[]
    label_list=[]
    dir_counter=0
    IMG_SIZE=128

    #对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
    for child_dir in os.listdir(path):
        child_path=os.path.join(path, child_dir)
        for dir_image in os.listdir(child_path):
            if endwith(dir_image,'jpg'):
                img=cv2.imread(os.path.join(child_path, dir_image))
                resized_img=cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                recolored_img=cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
                img_list.append(recolored_img)
                label_list.append(dir_counter)

        dir_counter+=1  #索引自增

    img_list=np.array(img_list)# 返回的img_list转成了 np.array的格式

    return img_list,label_list,dir_counter


def read_name_list(path):
    #读取训练数据集的文件夹，把他们的名字返回给一个list
    name_list=[]
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


class Model(object):
    #建立一个基于CNN的人脸识别模型
    FILE_PATH="face.h5"
    # 模型进行存储和读取的地方
    IMAGE_SIZE=128
    # 模型接受的人脸图片一定得是128*128的

    def __init__(self):
        self.model=None


    # 读取实例化后的DataSet类作为进行训练的数据源
    def read_trainData(self,dataset):
        self.dataset=dataset

    # 建立一个CNN模型，一层卷积、一层池化、一层卷积、一层池化、抹平之后进行全链接、最后进行分类
    def build_model(self):
        self.model = Sequential()
        #下面的代码都是按顺序添加CNN网络需要的各层，一个add就是一个网络曾
        self.model.add(
            Convolution2D(   #2维卷积层
                filters=32,
                kernel_size=(5, 5),
                padding='same',
                dim_ordering='th',
                input_shape=self.dataset.X_train.shape[1:]
            )
        )

        self.model.add(Activation('relu'))  #激活函数层
        self.model.add(
            MaxPooling2D(   #池化层
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )
        self.model.add(Convolution2D(filters=64, kernel_size=(5,5), padding='same')) #2维卷积层
        self.model.add(Activation('relu'))   #激活函数层
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')) #池化层
        self.model.add(Flatten())   #flatten
        self.model.add(Dense(1024))   #Dense层，又被称作全连接层
        self.model.add(Activation('relu'))   #激活函数层
        self.model.add(Dense(self.dataset.num_classes))    #dense
        self.model.add(Activation('softmax'))   #分类层，输出最终结果

        #输出模型概况
        self.model.summary()

    # 进行模型训练的函数，对具体的optimizer、loss进行不同选择
    def train_model(self):
        self.model.compile(
            optimizer='adam',  #可选的optimizer，例如RMSprop,Adagrad，个人感觉差异不大
            loss='categorical_crossentropy',   #也可以选用squared_hinge作为loss
            metrics=['accuracy'])

        # epochs、batch_size为可调的参数，epochs为训练多少轮、batch_size为每次训练多少个样本
        self.model.fit(self.dataset.X_train,self.dataset.Y_train,epochs=10,batch_size=10)


    def evaluate_model(self):
        print('\nTesting---------------')
        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)
        print('test loss;', loss)
        print('test accuracy:', accuracy)


    def save(self, file_path=FILE_PATH):
        print('Model Saved Finished!!!')
        self.model.save(file_path)


    def load(self, file_path=FILE_PATH):
        print('Model Loaded Successful!!!')
        self.model = load_model(file_path)

    # 需要确保输入的img得是灰化之后（channel =1 )且 大小为IMAGE_SIZE的人脸图片
    def predict(self,img):
        img=img.reshape((1, 1, self.IMAGE_SIZE, self.IMAGE_SIZE))
        # 像素数据浮点化以便归一化
        img=img.astype('float32')
        # 将其归一化,图像的各像素值归一化到0~1区间
        img=img/255.0
        result=self.model.predict_proba(img)    #测算一下该img属于某个label的概率
        max_index=np.argmax(result)    #找出概率最高的
        return max_index,result[0][max_index]   #第一个参数为概率最高的label的index,第二个参数为对应概率

#训练及评估方式
if __name__ == '__main__':
    dataset=DataSet('dataset/')
    model=Model()
    model.read_trainData(dataset)
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save()


