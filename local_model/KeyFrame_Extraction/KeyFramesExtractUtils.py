# -*- coding: utf-8 -*-
"""
this key frame extract algorithm is based on interframe difference.
The principle is very simple
First, we load the video and compute the interframe difference between each frames
Then, we can choose one of these three methods to extract keyframes, which are
all based on the difference method:
1. use the difference order
    The first few frames with the largest average interframe difference
    are considered to be key frames.
2. use the difference threshold
    The frames which the average interframe difference are large than the
    threshold are considered to be key frames.
3. use local maximum
    The frames which the average interframe difference are local maximum are
    considered to be key frames.
    It should be noted that smoothing the average difference value before
    calculating the local maximum can effectively remove noise to avoid
    repeated extraction of frames of similar scenes.
After a few experiment, the third method has a better key frame extraction effect.
"""
import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from scipy.signal import argrelextrema
 
 
# USE_LOCAL_MAXIMA=True，效果还行，时间有点长，54张图，包括一张差异趋势图
# 使用cv2.CAP_PROP_POS_FRAMES间隔取帧，主要消耗CPU，效率比内存高，效果更好
 
# 视频提取关键帧工具类
class KeyFramesExtractUtils:
 
    # 初始化
    def __init__(self, video_path=None, save_path=None):
        self.video_path = video_path
        self.save_path = save_path
 
    # 提取关键帧
    def extract_keyframe(self, method="use_local_maxima"):
        # print(sys.executable)
        # print("method===>", method)
 
        # fixed threshold value
        thresh = 0.6
 
        # Number of top sorted frames
        num_top_frames = 50
 
        # smoothing window size
        len_window = int(50)
 
        # print("target video: " + self.video_path)
        # print("frame save directory: " + self.save_path)
        # load video and compute diff between frames
        cap = cv2.VideoCapture(str(self.video_path))
        curr_frame = None
        prev_frame = None
        frame_diffs = []
        frames = []
        video_frames = []
        key_frame = 10  # 所隔帧数
 
        t0_start = time.time()
 
        k = 0
        j = 0
        while True:
            if key_frame >= 1:
                t1_start = time.time()
                # print("j======>", j)
                cap.set(cv2.CAP_PROP_POS_FRAMES, j)  # 这里的cv2.CAP_PROP_POS_FRAMES参数是说：取第j帧之后的那一帧
                j += key_frame
                success, frame = cap.read()
 
                if not success:
                    # print("第一次视频帧读取完毕!")
                    break
 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                curr_frame = gray
                if curr_frame is not None and prev_frame is not None:
                    # logic here
                    diff = cv2.absdiff(curr_frame, prev_frame)
                    diff_sum = np.sum(diff)
                    diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
 
                    frame_diffs.append(diff_sum_mean)
                    temp_frame = Frame(k, diff_sum_mean)
                    frames.append(temp_frame)
                    video_frames.append(frame)
                    k = k + 1
                prev_frame = curr_frame
 
                t1_end = time.time()
                # print("计算一次差分耗时：", (t1_end - t1_start), "s")
 
        cap.release()
        t0_end = time.time()
        # print("计算共耗时：", (t0_end - t0_start), "s")
 
        t_start = time.time()
 
        # compute keyframe
        keyframe_id_set = set()
        if method == "use_top_order":
            print("---------------Using use_top_order---------------")
            # sort the list in descending order
            frames.sort(key=operator.attrgetter("diff"), reverse=True)
            for keyframe in frames[:num_top_frames]:
                keyframe_id_set.add(keyframe.id)
        elif method == "use_thresh":
            print("---------------Using Threshold---------------")
            for i in range(1, len(frames)):
                if rel_change(np.float(frames[i - 1].diff), np.float(frames[i].diff)) >= thresh:
                    keyframe_id_set.add(frames[i].id)
        else:
            print("---------------Using Local Maxima---------------")
            diff_array = np.array(frame_diffs)
            sm_diff_array = smooth(diff_array, len_window)
            frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
            for i in frame_indexes:
                keyframe_id_set.add(frames[i - 1].id)
 
            plt.figure(figsize=(40, 20))
            # Set the number of ticks on the x-axis
            plt.xticks(range(len(sm_diff_array)))
            plt.stem(sm_diff_array)
            plt.savefig(self.save_path + 'plot.png')
        keyFrame_file_path=[]
        for idx in keyframe_id_set:
            name = "keyframe_" + str(idx) + ".jpg"
            cv2.imwrite(self.save_path + name, video_frames[idx])
            keyFrame_file_path.append([idx, f"{self.save_path}/{name}"])
            # print(len(keyFrame_file_path))
        # t_end = time.time()
        # print("取帧共耗时：", (t_end - t_start), "s")
        return keyFrame_file_path
 
 
class Frame:
    """class to hold information about each frame
    """
 
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff
 
    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id
 
    def __gt__(self, other):
        return other.__lt__(self)
 
    def __eq__(self, other):
        return self.id == other.id and self.id == other.id
 
    def __ne__(self, other):
        return not self.__eq__(other)
 
 
def rel_change(a, b):
    x = (b - a) / max(a, b)
    # print(x)
    return x
 
 
def smooth(x, window_len=13, window='hanning'):
    """smooth the data using a window with requested size.
    """
    # print(len(x), window_len)
    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    # print(len(s))
 
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]
 
 
if __name__ == "__main__":
    keyFrame = KeyFramesExtractUtils(video_path="/home/nkd/Documents/ssd_nvme0n1/jiajiyuan/ai_agent/Facial_Expression_Recognition/data/test_video/致新书院2023年植树节活动.m4v", save_path="./")
    keyFrame.extract_keyframe(method="use_local_maxima")
