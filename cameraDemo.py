#!usr/bin/env python
#encoding:utf-8
from __future__ import division


#功能： 人脸识别摄像头视频流数据实时检测模块


import os
import cv2
from faceRegnigtionModel import Model


threshold=0.7   #如果模型认为概率高于70%则显示为模型中已有的人物


def read_name_list(path):
    # 读取训练数据集
    name_list=[]
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


class Camera_reader(object):
    # 在初始化camera的时候建立模型，并加载已经训练好的模型，我在加载视像头时经常卡。
    def __init__(self):
        self.model=Model()
        self.model.load()
        self.img_size=128


    def build_camera(self):
       # 调用摄像头来实时人脸识别
        face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')
        # opencv人脸检测的级联分类器，用于帮助识别图像或者视频流中的人脸
        name_list=read_name_list('dataset/')   #读取dataset中标签看是否匹配
        cameraCapture=cv2.VideoCapture(0)    #N为整数内置摄像头为0，若有其他摄像头则依次为1，2......
        success, frame=cameraCapture.read()
        while success and cv2.waitKey(1)==-1:       #获取用户输入，同时可获取按键的ASCLL码值
            success,frame=cameraCapture.read()
            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    #将图片进行转灰，，降低计算复杂度
            faces=face_cascade.detectMultiScale(gray, 1.3, 5)     #识别人脸
            for (x,y,w,h) in faces:
                ROI=gray[x:x+w,y:y+h]
                ROI=cv2.resize(ROI, (self.img_size, self.img_size),interpolation=cv2.INTER_LINEAR)
                label,prob=self.model.predict(ROI)   #利用模型对cv2识别出的人脸进行比对
                if prob>threshold:   
                    show_name=name_list[label]   #大于设定的识别概率，输出该标签
                else:
                    show_name="Stranger"     #未与标签成功比对，为stranger
                cv2.putText(frame, show_name, (x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)    #显示名字
                frame=cv2.rectangle(frame,(x,y), (x+w,y+h),(255,0,0),2)    #在人脸区域画一个正方形出来
            cv2.imshow("Camera", frame)  #展示识别窗口

         #释放摄像头并销毁所有窗口
        cameraCapture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    camera=Camera_reader()
    camera.build_camera()


