import cv2 as cv
import os
import numpy as np


IMG_SAVE_PATH = 'face/3/'
# 读取所有图片路径
image_paths = [os.path.join(IMG_SAVE_PATH, f) for f in os.listdir(IMG_SAVE_PATH)]

# 读取所有图片并转成灰度图像
faces_list = []
for image_path in image_paths:
    img = cv.imread(image_path, 0)
    if img is not None:
        faces_list.append(img)

# 读取所有标签
labels = []
for f in os.listdir(IMG_SAVE_PATH):
    try:
        label = int(f.split('.')[0])
        labels.append(label)
    except ValueError:
        continue

# 加载识别器
recognizer = cv.face.LBPHFaceRecognizer_create()

# 训练数据，放入人脸信息列表和标签数组
if len(faces_list) > 0 and len(labels) > 0:
    recognizer.train(faces_list, np.array(labels))
else:
    print("没有足够的人脸数据或标签进行训练，请检查数据。")

# 保存数据
if len(faces_list) > 0 and len(labels) > 0:
    recognizer.write('train.yml')