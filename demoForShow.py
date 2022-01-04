#!usr/bin/env python
#encoding:utf-8
from __future__ import division


#功能： 将人脸识别模型暴露为web服务接口，用于演示的demo


import os
import cv2
import sys
import time
import numpy as np
from flask import Flask
from flask import request
from faceRegnigtionModel import Model
app=Flask(__name__)


def endwith(s,*endstring):
    #对字符串的后续和标签进行匹配
    resultArray=map(s.endswith,endstring)
    if True in resultArray:
        return True
    else:
        return False


def read_file(path):
   #图片读取
    img_list=[]
    label_list=[]
    dir_counter=0
    IMG_SIZE=128     #图片大小必须为128*128
    for child_dir in os.listdir(path):
        child_path=os.path.join(path, child_dir)
        for dir_image in os.listdir(child_path):
            if endwith(dir_image,'jpg'):
                #从初始路径开始叠加，合成可识别的操作路径
                img=cv2.imread(os.path.join(child_path, dir_image))
                resized_img=cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                #图片大小统一标准化
                recolored_img=cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
                # 将图片进行转灰
                img_list.append(recolored_img)
                label_list.append(dir_counter)

        dir_counter+=1  #索引自增

    img_list=np.array(img_list)   #图片存为一个List

    return img_list,label_list,dir_counter


def read_name_list(path):
    #读取训练集
    name_list=[]
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


def detectOnePicture(path):
    #对单张图片进行识别
    model=Model()
    model.load()
    img=cv2.imread(path)
    img=cv2.resize(img,(128,128))
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    picType,prob=model.predict(img)
    if picType!=-1:
        name_list=read_name_list('dataset/')
        print(name_list[picType],prob)
        res=u"识别为： "+name_list[picType]+u"的概率为： "+str(prob)
    else:
        res=u"未识别出该人！尝试增加图片来训练模型！"
    return res


@app.route("/")
def init():
    #初始化接口
    return u"人脸识别程序正常启动！"


@app.route("/detect", methods=["GET"])
def detectFace():
    #人脸识别接口
    if request.method=="GET": 
        picture=request.args['picture']
    start_time=time.time()
    res=detectOnePicture(picture)
    end_time=time.time()
    execute_time=str(round(end_time-start_time,2))
    tsg=u' 总耗时为： %s 秒' % execute_time
    msg=res+'\n\n'+tsg
    return msg


if __name__ == "__main__":
    print('faceRegnitionDemo')
    #自己的IP地址，端口号
    app.run(host='127.0.0.1',port=5000)