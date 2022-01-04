# Face-recognition
Face-recognition
1.1
完整地从数据收集，数据预处理，CNN卷积神经网络搭建，模型训练与测试和最终效果使用完成了一个简单的人脸检测与识别系统设计，具体地了解人脸检测与识别的设计方法工作流程，尤其是亲自搭建keras下的神经网络，对神经网络各层有了进一步的学习与理解.

1.2 项目文件说明
文件名	实现功能
Config	Opencv的分类器配置文件
Data	网络公开的数据集下载，个人照片的添加
Dataset	Data文件中的数据经过标准化处理后存储的目录路径
Pics	结果截图
getCamaraPics.py	基于摄像头视频流数据获取人脸图片并存储

dataHelper.py
	原始图像数据的预处理工作，负责将原始图像转化为标准数据文件

faceRegnigtionModel.py
	人脸识别模块，负责构建模型并保存本地，默认为face.h5

demoForShow.py
	人脸识别展示Demo模块，通过将模型的调用暴露成web服务，实现在浏览器端进行展示

cameraDemo.py
	调用摄像头来进行实时的人脸数据识别

1.3 项目环境配置
环境	版本	功能说明
Windows	10	在windows环境下进行实现
Python	3.6	Python语言简单，可引用依赖库，功能强大
Opencv	4.2.0	包含丰富计算机视觉算法，用于图像处理分析
Tensorflow	1.14.0	广泛使用的机器学习和数学运算的算法库
Numpy	1.18.3	存储和处理矩阵，支持维度数组与矩阵计算
Keras	2.2.4	深度学习模型的设计、调式、评估、应用
Flask	1.1.2	Web应用框架
Skleran	0.0	高效实现算法的工具包
Cuda	10.0.0	支持GPU高性能计算

1.4 需要将Data和dataset文件夹解压即可

1.5 运行训练和测试会得到face.h5模型文件，此处因为文件太大（195Mb） 没有上传。
