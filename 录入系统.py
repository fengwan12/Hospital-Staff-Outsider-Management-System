import cv2 as cv
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
import io


def main():
    # 打开摄像头
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('摄像头连接失败')

    # 输入姓名
    cv.waitKey(100)  # 短暂停顿，让 cv2 相关窗口初始化完成，让出输入焦点
    name = input('请输入需要录入的名字：')
    print("姓名输入完成，按's'键保存图片，按'q'键退出")

    # 循环读取每一帧画面
    while True:
        ret, frame = cap.read()
        if not ret:
            print('读帧失败')
            break

        # 提取的人脸信息和灰度图像
        faces, gray = img_extract_faces(frame)

        # 框出人脸
        for x, y, w, h in faces:
            cv.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)

        # 展示人脸
        frame = cv2pil(frame)
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype("STKAITI.TTF", 20)  # 指定字体和大小
        draw.text((10, 10), f'姓名：{name}', font=font, fill=(255, 0, 0))
        frame = pil2cv2(frame)

        cv.imshow('face', frame)

        k = cv.waitKey(1)

        # 按 's' 键保存人脸
        if k == ord('s'):
            save_face(faces, frame, name)

        # 按 'q' 键退出
        elif k == ord('q'):
            break

    # 关闭摄像头和窗口
    cap.release()
    cv.destroyAllWindows()


def img_extract_faces(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_classifier = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = face_classifier.detectMultiScale(gray)
    return faces, gray


# 保存提取到的人脸信息和姓名
def save_face(faces, img, name):
    if len(faces) == 0:
        print('没有检测到人脸，请调整')
        return
    if len(faces) > 1:
        print('检测到多个人脸，请调整')
        return
    x, y, w, h = faces[0]
    cv.imwrite(get_img_name(name), img[y:y + h, x:x + w])
    print('录入成功，按 q 键退出')


# 返回保存路径 + name_number + '.' + name + '.jpg'
def get_img_name(name):
    # 读取保存路径下所有的目录，格式：name_number + '.' + name + '.jpg'
    name_map = {f.split('.')[1]: int(f.split('.')[0]) for f in os.listdir(IMG_SAVE_PATH)}
    if not name_map:
        name_number = 1
    elif name in name_map:
        name_number = name_map[name]
    else:
        name_number = max(name_map.values()) + 1
    return IMG_SAVE_PATH + str(name_number) + '.' + name + '.jpg'


def cv2pil(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return Image.fromarray(image)


def pil2cv2(image):
    # 将 PIL 图像转换为 numpy 数组
    image = np.array(image)
    # 使用 cv2 的 cvtColor 函数将 RGB 转换为 BGR
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image



if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    IMG_SAVE_PATH = 'face/3/'
    main()