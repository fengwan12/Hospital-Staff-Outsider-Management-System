#  行人检测系统

import cv2
import torch
import numpy as np


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


if __name__ == "__main__":
    detect_and_show_video()