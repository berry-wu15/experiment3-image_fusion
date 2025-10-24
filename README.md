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
#图像a转灰度：cv2.COLOR_BGR2GRAY将BGR格式转为灰度格式
gray_a = cv2.cvtColor(img_a,cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2GRAY)
```
