import cv2
import os
import numpy as np
from datetime import datetime
import PySimpleGUI as sg
from PIL import Image, ImageDraw, ImageFont # type: ignore
import sys
import io
import torch # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.font_manager as fm # type: ignore
import subprocess

# 获取当前日期并生成签到记录文件路径
today = datetime.now().strftime("%Y%m%d")
SIGNED_IN_FILE = os.path.join('face/2/', f'signed_in_{today}.txt')
# 保存人脸图像的目录
IMG_SAVE_PATH = 'face/3/'

# 职位和部门选项
positions = ['doctor', 'Specialist', 'Resident','Intern','Nurse','Anesthesiologist','Hospital Director','Administrator']  # 示例职位
departments = ['Emergency Department', 'Outpatient Department', 'Inpatient Department','Internal Medicine','Surgery','Obstetrics and Gynecology','Pediatrics','Otorhinolaryngology','Dentistry','Dermatology']  # 示例部门

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


def main(name, position, department):
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('摄像头连接失败')
        return

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
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)

        # 展示人脸
        frame = cv2pil(frame)
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype("STKAITI.TTF", 20)
        draw.text((10, 10), f'姓名：{name} 职位：{position} 部门：{department}', font=font, fill=(255, 0, 0))
        frame = pil2cv2(frame)

        cv2.imshow('face', frame)

        k = cv2.waitKey(1)

        # 按 's' 键保存人脸
        if k == ord('s'):
            save_face(faces, frame, f'{name}_{position}_{department}')

        # 按 'q' 键退出
        elif k == ord('q'):
            break

    # 关闭摄像头和窗口
    cap.release()
    cv2.destroyAllWindows()
       # 读取所有图片路径
    image_paths = [os.path.join(IMG_SAVE_PATH, f) for f in os.listdir(IMG_SAVE_PATH)]

    # 读取所有图片并转成灰度图像
    faces_list = []
    for image_path in image_paths:
        img = cv2.imread(image_path, 0)
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
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # 训练数据，放入人脸信息列表和标签数组
    if len(faces_list) > 0 and len(labels) > 0:
        recognizer.train(faces_list, np.array(labels))
    else:
        print("没有足够的人脸数据或标签进行训练，请检查数据。")

    # 保存数据
    if len(faces_list) > 0 and len(labels) > 0:
        recognizer.write('train.yml')
def img_extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
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
    cv2.imwrite(get_img_name(name), img[y:y + h, x:x + w])
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def pil2cv2(image):
    # 将 PIL 图像转换为 numpy 数组
    image = np.array(image)
    # 使用 cv2 的 cvtColor 函数将 RGB 转换为 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

 


def identity_verification_system():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('train.yml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('摄像头连接失败')
        return

    name_map = {int(f.split('.')[0]): f.split('.')[1] for f in os.listdir(IMG_SAVE_PATH)}

    while True:
        ret, frame = cap.read()
        if not ret:
            print('读帧失败')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            img_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence > 85:
                name = 'unknown'
            else:
                name = name_map[img_id]
            cv2.putText(frame, name, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0, 255, 0), thickness=2)
            cv2.circle(frame, (x + w // 2, y + h // 2), radius=w // 2, color=(255, 0, 0), thickness=2)

        cv2.imshow('face', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_and_show_video():
    # 加载 YOLOv5 模型
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 使用 YOLOv5s 模型
    print("类别：", model.names)

    # 打开视频流
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: 无法打开监控摄像头，请检查。")
        exit(1)
    print("监控摄像头成功打开。")

    # 获取原始视频的帧率、宽度和高度
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 存储行人数量的列表
    people_counts = []
    current_people = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_image)
        boxes = results.xyxy[0].cpu().numpy()
        current_people = 0  # 重置当前人数
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            if cls == 0:  # 假设类别0是行人
                current_people += 1  # 增加当前人数
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Current People: {current_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Video with Detection', frame)
        people_counts.append(current_people)
        if cv2.waitKey(int(1000/fps)) & 0xFF == 27:  # 按ESC键退出
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    # 将人流量数据保存到文件
    np.savetxt('people_counts.txt', people_counts, fmt='%d')
    print("人流量数据已保存到 people_counts.txt 文件中。")

def plot_people_count_from_file():
     # 检查文件是否存在
    if not os.path.exists('people_counts.txt'):
        print("文件 people_counts.txt 不存在，请检查文件路径。")
        exit(1)
    # 从文件中读取人流量数据
    people_counts = np.loadtxt('people_counts.txt', dtype=int)
    # 假设文件中的一个数字代表一帧，帧率为 30 帧/秒，将帧转换为秒
    fps = 30
    times = np.arange(len(people_counts)) / fps  

    # 处理字体显示问题，使用系统中的中文字体
    font_path = fm.findfont(fm.FontProperties(family='SimHei'))
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False  

    # 绘制图表
    fig, ax = plt.subplots()
    ax.plot(times, people_counts)
    ax.set_xlabel('时间（秒）')
    ax.set_ylabel('人数')
    ax.set_title('人流量变化')
    plt.show(block=True)


# 创建 PySimpleGUI 布局
layout = [
    [sg.Button('录入人像系统'),  sg.Button('身份验证系统'),  sg.Button('员工打卡'),sg.Button('行人检测并查看行人流量图')],
    [sg.Text('', size=(40, 1), key='-OUTPUT-')]
]

window = sg.Window('用户交互界面', layout)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    elif event == '录入人像系统':
        # 创建一个新的窗口用于输入姓名、选择职位和部门
        input_layout = [
            [sg.Text('请输入姓名：'), sg.InputText(key='-NAME-')],
            [sg.Text('请选择职位：'), sg.Combo(positions, default_value=positions[0], key='-POSITION-')],
            [sg.Text('请选择部门：'), sg.Combo(departments, default_value=departments[0], key='-DEPARTMENT-')],
            [sg.Button('确定'), sg.Button('取消')]
        ]
        input_window = sg.Window('输入姓名、职位、部门信息', input_layout)
        input_event, input_values = input_window.read()
        if input_event == '确定':
            name = input_values['-NAME-']
            position = input_values['-POSITION-']
            department = input_values['-DEPARTMENT-']
            if name and position and department:
                window['-OUTPUT-'].update('开始录入人像系统...')
                main(name, position, department)
                window['-OUTPUT-'].update('录入人像系统结束')
            else:
                sg.popup('请输入姓名、选择职位和部门')
        input_window.close()
    elif event == '身份验证系统':
        window['-OUTPUT-'].update('正在运行身份验证...')
        identity_verification_system()
        window['-OUTPUT-'].update('身份验证完成')
    elif event == '行人检测并查看行人流量图':
        window['-OUTPUT-'].update('正在绘制行人流量图...')
        # 创建并启动视频检测进程
        video_process = subprocess.Popen([sys.executable, '-c', 'import sys; from xingren import detect_and_show_video; detect_and_show_video()'])
        # 确保视频检测进程完成后再进行绘图
        video_process.communicate()
        plot_people_count_from_file()
        window['-OUTPUT-'].update('行人流量图绘制完成')
    elif event == '员工打卡':
        window['-OUTPUT-'].update('正在进行员工打卡...')
        sign_in()
        window['-OUTPUT-'].update('员工打卡完成')

window.close()