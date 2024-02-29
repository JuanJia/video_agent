'''
将复杂任务拆分成简单的原子任务，写入CSV文件
'''

# import openpyxl
# from IPython.display import display, Image, Audio

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
# import requests
# import pandas as pd
import json
import io
import random
import csv
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numbers
import torchvision
import torch
import numpy as np
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
import imageio
from tqdm import tqdm

# from openpyxl.utils.dataframe import dataframe_to_rows

# os.environ["http_proxy"] = "http://localhost:33333"
# os.environ["https_proxy"] = "http://localhost:33333"

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:33210"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:33210"

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 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
# API_SECRET_KEY = "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"
# API_SECRET_KEY = "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"

# API_SECRET_KEY = "sk-fsmITOyCiZeFCtOFE20aBa3b45624a5dA9F5Bc7e3e6b8aD4"
# API_SECRET_KEY = "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"
API_SECRET_KEY = "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"

# BASE_URL = "https://flag.smarttrot.com/v1/"  #智增增的base_url
# BASE_URL = "https://oneapi.xty.app"  #智增增的base_url
BASE_URL="https://api.openai.com/v1/chat/completions"

# API_SECRET_KEY = "zk-a6c32f0dbc0afadd9ef8b505b1f81bdb"

vi=[]
ans=[]
apis=[]
api_score=[]

from openai import OpenAI

def gpt4v_video(base64Frames,q):
    total_frames = len(base64Frames)  # 获取列表总长度
    interval = total_frames // 10  # 计算间隔

    # 使用切片操作抽取帧
    selected_frames = base64Frames[::interval][:10]

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                q,
                *map(lambda x: {"image": x}, selected_frames),
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 4090,
    }

    try:
        result = client.chat.completions.create(**params)
        return result.choices[0].message.content
    except Exception as e:
        print(f"An exception occurred: {e}")
        return ""

data_list = {
    "Action Sequence": ("action_sequence.json", "/mnt/xuyibo/MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "/mnt/xuyibo/MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "/mnt/xuyibo/MVBench/video/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "/mnt/xuyibo/MVBench/video/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "/mnt/xuyibo/MVBench/video/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "/mnt/xuyibo/MVBench/video/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "/mnt/xuyibo/MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", "/mnt/xuyibo/MVBench/video/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "/mnt/xuyibo/MVBench/video/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "/mnt/xuyibo/MVBench/video/sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", "/mnt/xuyibo/MVBench/video/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "/mnt/xuyibo/MVBench/video/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "/mnt/xuyibo/MVBench/video/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "/mnt/xuyibo/MVBench/video/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "/mnt/xuyibo/MVBench/video/perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "/mnt/xuyibo/MVBench/video/nturgbd/", "video", False),
    "Character Order": ("character_order.json", "/mnt/xuyibo/MVBench/video/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "/mnt/xuyibo/MVBench/video/vlnqa/", "video", False),
    # "Episodic Reasoning": ("episodic_reasoning.json", "/mnt/xuyibo/MVBench/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Counterfactual Inference": ("counterfactual_inference.json", "/mnt/xuyibo/MVBench/video/clevrer/video_validation/", "video", False),
}

data_dir = "/mnt/xuyibo/MVBench/json"


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor
    
class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]
    
class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]
    
class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2)
                                   for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1]
                                       for x in img_group], axis=2)
            else:
                #print(np.concatenate(img_group, axis=2).shape)
                # print(img_group[0].shape)
                return np.concatenate(img_group, axis=2)
            
class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(
                    pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, num_segments=8, resolution=224):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        
        self.num_segments = num_segments
        
        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        # torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            # 'video': torch_imgs,
            'video': video_path,
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }

if __name__ == '__main__':
    dataset = MVBench_dataset(data_dir, data_list)
    client = OpenAI(api_key=API_SECRET_KEY)

    # 打开CSV文件，准备写入
    with open('output.csv', 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'video_path', 'question', 'response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入CSV文件的表头
        writer.writeheader()

        for example_id, example in enumerate(tqdm(dataset)):
            # print(example)
            if (example_id <= 3179):
                continue
            video_path = example["video"]
            video = cv2.VideoCapture(video_path)
            base64Frames = []
            while video.isOpened():
                success, frame = video.read()
                if not success:
                    break
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

            video.release()
            question = example['question']
            # print(len(base64Frames), "frames read.")
            q = "定义复杂任务在机器学习中可被定义为由多个原子任务组成的综合性问题，只需要一种原子任务即可完成的则不属于复杂任务。原子任务包含："\
                "（分类、目标检测（识别)、分割、目标追踪、边缘检测、姿势评估、理解CNN、图像降噪、超分辨率重建、序列学习、特征检测与匹配、图像标定，视频标定、问答系统、图片生成（文本生成图像）、视觉关注性和显著性（质量评价）、人脸识别、3D重建、推荐系统、细粒度图像分析、图像压缩。）\n"\
                f"问题:\"{question}\"，请判断回答以上引号包含的问题是否属于复杂任务。若属于，只需要给出完成所需要的多个原子任务；若不属于，只需要给出完成所需要的单个原子任务。你不需要直接回答引号包含的问题。完成所需要的原子任务有："
            response = gpt4v_video(base64Frames, q)
            # print(response)

            # 将结果写入CSV文件
            writer.writerow({
                'id': example_id,
                'video_path': video_path,
                'question': question,
                'response': response
            })
