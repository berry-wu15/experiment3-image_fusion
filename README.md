# experiment3-image_fusion(实验三-图像融合)
This experiment achieves image stitching based on SIFT feature matching and perspective transformation.

这个实验基于SIFT特征匹配和透视变换完成图像拼接

## 1.Experimental Purpose
To stitch two images with overlapping regions into a complete image by using SIFT feature matching and perspective transformation techniques, and verify the effectiveness of the image stitching process.

利用SIFT特征匹配和透视变换技术，将具有重叠区域的两幅图像拼接成一幅完整图像，并验证图像拼接过程的有效性。

## 2.Experimental Content

#### 2.1 Image Reading and Preprocessing

图像读取与预处理

Read the two images to be stitched and convert them into grayscale images to prepare for subsequent feature extraction

读取待拼接的两张图像，并将其转为灰度图，为后续特征提取做准备

```
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
```
img_a = cv2.imread(r"C:\Users\Wxy\Desktop\computer-vision\3-1.jpg")
img_b = cv2.imread(r"C:\Users\Wxy\Desktop\computer-vision\3-2.jpg")
```
```
#图像a转灰度：cv2.COLOR_BGR2GRAY将BGR格式转为灰度格式`#hsl(39,100%,50%)`
gray_a = cv2.cvtColor(img_a,cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2GRAY)
```

#### 2.2 SIFT Feature Extraction and Matching

SIFT特征提取与匹配

Initialize the SIFT detector to extract feature points and descriptors from the two grayscale images.Use the FLANN matcher to match the features,and then filter out high-quality matching points through Lowe's ratio test.

初始化 SIFT 检测器，提取两张灰度图的特征点与描述符，用 FLANN 匹配器匹配特征，再通过 Lowe's 比率测试筛选优质匹配点

```
#初始化SIFT检测器并提取特征
sift = cv2.SIFT_create()
#提取图像a的特征：detectAndCompute()返回关键点（kp_a）和描述符（des_a），None表示不使用掩码
kp_a,des_a = sift.detectAndCompute(gray_a,None)   # 图像a的关键点（位置、尺度等）和描述符（128维向量）
kp_b,des_b = sift.detectAndCompute(gray_b,None)   #图像b的关键点和描述符
```
```
#使用FLANN匹配器进行特征匹配
# 使用FLANN匹配器进行特征匹配：FLANN（快速最近邻搜索库）比暴力匹配更快，适合大量特征点匹配
# 定义FLANN匹配器的算法类型：K-D树算法（适合高维数据匹配）
FLANN_INDEX_KDTREE = 1
#配置FLANN的索引参数：algorithm指定算法类型，trees=5表示构建5棵K-D树（平衡速度与精度）
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
#配置FLANN的搜索参数：checks=50表示每个特征点最多检查50个邻居，次数越多匹配越准但速度越慢
search_params = dict(checks=50)  #检查次数越多匹配越准确但速度更慢
#创建FLANN匹配器实例：传入索引参数和搜索参数
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des_a,des_b,k=2)  #k=2表示每个特征点返回2个最佳匹配
```
```
#应用Low's比率测试筛选优劣匹配点
good_matches = []
for m,n in matches:    
#Lowe's比率测试：若第一匹配距离 < 0.7*第二匹配距离，判定为优质匹配（0.7是经验阈值，可在0.7-0.8调整）    
if m.distance <0.6 * n.distance:
        good_matches.append(m)
```

#### 2.3 
```
#绘制匹配的SIFT关键点，绘制匹配点：drawMatches()将两张图像的关键点与匹配线绘制在同一张图中
matched_keypoints_img = cv2.drawMatches(
    img_a,kp_a,img_b,kp_b,good_matches,    
None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
```
```
#提取匹配点的坐标
#提取图像b的匹配点坐标：m.trainIdx是b中与a匹配的特征点索引，pt是关键点的(x,y)坐标；
#转换为float32类型并reshape为(-1,1,2)（OpenCV透视变换要求的输入格式：N个点，每个点为1×2向量）
src_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)#图像b的关键点
dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)#图像a的关键点
```






