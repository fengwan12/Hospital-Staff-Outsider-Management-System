import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm


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


if __name__ == "__main__":
    plot_people_count_from_file()