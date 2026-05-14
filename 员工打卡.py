import cv2
import os
import numpy as np

from datetime import datetime


# 获取当前日期并生成签到记录文件路径
today = datetime.now().strftime("%Y%m%d")
SIGNED_IN_FILE = os.path.join('face/2/', f'signed_in_{today}.txt')
# 保存人脸图像的目录
IMG_SAVE_PATH = 'face/3/'
# 签到记录文件路径
def sign_in():
    """
    签到函数
    读取已保存的人脸图像进行训练，打开摄像头识别人员并签到，显示签到状态
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('train.yml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('摄像头连接失败')
        return

    name_map = {int(f.split('.')[0]): f.split('.')[1] for f in os.listdir(IMG_SAVE_PATH) if f.endswith('.jpg')}
    signed_in_names = load_signed_in_names()

    # 设定一个固定的未知人脸判断阈值，这里通过实验调整为80
    unknown_face_threshold = 80

    while True:
        ret, frame = cap.read()
        if not ret:
            print('读帧失败')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        faces = face_classifier.detectMultiScale(gray)
        current_time = datetime.now()  # 获取当前时间
        # 在左上角显示当前时间
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        for x, y, w, h in faces:
            img_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence > unknown_face_threshold:
                name = 'Visitor'
                display_text = f'{name}'
            else:
                name = name_map[img_id]
                if name not in signed_in_names:
                    save_signed_in_name(name, current_time)  # 保存签到人员姓名和当前时间
                    signed_in_names.add(name)
                    display_text = f'{name} OK {current_time}'  # 显示签到成功和当前时间
                else:
                    display_text = f'{name} OK {current_time}'
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.circle(frame, (x + w // 2, y + h // 2), w // 2, (255, 0, 0), 2)
        cv2.imshow('face', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def load_signed_in_names():
    """
    加载已签到人员名单
    :return: 已签到人员集合
    """
    if not os.path.exists(SIGNED_IN_FILE):
        print(f"签到文件 {SIGNED_IN_FILE} 不存在")
        return set()
    try:
        with open(SIGNED_IN_FILE, 'r') as f:
            signed_in_names = set(line.strip() for line in f.readlines())
            print(f"成功加载签到人员名单: {signed_in_names}")
            return signed_in_names
    except Exception as e:
        print(f"加载已签到人员名单时出错: {e}")
        return set()


def save_signed_in_name(name, current_time):  # 增加当前时间参数
    """
    保存签到人员姓名
    :param name: 姓名
    :param current_time: 当前时间
    """
    try:
        with open(SIGNED_IN_FILE, 'a') as f:
            f.write(f'{name},{current_time}\n')  # 保存姓名和当前时间，用逗号分隔
            print(f"成功将 {name} 写入签到文件")
    except Exception as e:
        print(f"写入签到文件时出错: {e}")


if __name__ == '__main__':
        sign_in()
