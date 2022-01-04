#!usr/bin/env python
#encoding:utf-8 编码方式说明

'''
功能：通过opencv调用摄像头获取个人图片，用以后续训练学习，以及是否为使用者的检测与识别
内容：启动摄像头后，通过借助键盘输入操作来完成图片的获取工作
        c: 生成存储目录
        p: 执行截图
        q: 退出拍摄
'''
 
 
import os
import cv2
import sys





#调用电脑摄像头来自动获取图片
def cameraAutoForPictures(saveDir='data/'):
    #图片存储位置
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    count=1  #图片计数索引
    cap=cv2.VideoCapture(0)   #N为整数内置摄像头为0，若有其他摄像头则依次为1，2......
    width,height,w=640,480,360
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)  #设置capture窗口参数
    crop_w_start=(width-w)//2
    crop_h_start=(height-w)//2
    print('width: '),width
    print('height: '),height
    while True:
        ret,frame=cap.read()  #读取图像
        frame=frame[crop_h_start:crop_h_start+w,crop_w_start:crop_w_start+w]   #展示相框
        frame=cv2.flip(frame,1,dst=None)  #前置摄像头获取的画面是非镜面的，即左手会出现在画面的右侧，此处使用flip进行水平镜像处理
        cv2.imshow("capture", frame)
        action=cv2.waitKey(1) & 0xFF    #获取用户输入，同时可获取按键的ASCLL码值
        if action==ord('c'):   #输入新的存储目录
            saveDir=raw_input(u"请输入新的存储目录：")
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
        elif action==ord('p'):  #拍摄下一张图片
            cv2.imwrite("%s/%d.jpg" % (saveDir,count),cv2.resize(frame, (224, 224),interpolation=cv2.INTER_AREA))
            print(u"%s: %d 张图片" % (saveDir,count))
            count+=1
        if action==ord('q'):  #退出
            break
    cap.release()  #释放摄像头
    cv2.destroyAllWindows()  #丢弃窗口
 
 
if __name__=='__main__':   
    #xxx可替换修改，表示拍摄的单个人的不同姿态照片，文件名视为标签，最后的识别为此名
    cameraAutoForPictures(saveDir='data/KK/')