"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""

import re
from unidecode import unidecode  # 需要安装 unidecode 库，用于转换非ASCII字符
import argparse
import os

from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, default_conversation, conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

import base64
import io
import random
import csv
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

class MMBenchDataset(Dataset):
    def __init__(self,
                 data_file,
                 sys_prompt='There are several options:'):
        self.df = pd.read_csv(data_file, sep='\t')
        self.sys_prompt = sys_prompt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        image = decode_base64_to_image(image)
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None
        catetory = self.df.iloc[idx]['category']
        l2_catetory = self.df.iloc[idx]['l2-category']

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }
        options_prompt = f'{self.sys_prompt}\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        hint = self.load_from_df(idx, 'hint')
        data = {
            'img': image,
            'question': question,
            'answer': answer,
            'options': options_prompt,
            'category': catetory,
            'l2-category': l2_catetory,
            'options_dict': options,
            'index': index,
            'context': hint,
        }
        return data
    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None
        
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/video_llama_eval_withaudio.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='LLama2', help="The type of LLM")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


class ChatBot:

    def __init__(self, args):
        self.chat = self._init_model(args)
        if args.model_type == 'vicuna':
            self.chat_state = default_conversation.copy()
        else:
            self.chat_state = conv_llava_llama_2.copy()
        self.img_list = list()
        self.set_para()

    def _init_model(self, args):
        print('Initializing Chat')
        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
        model.eval()
        vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        print('Initialization Finished')
        return chat

    def set_para(self, num_beams=1, temperature=0.2):
        self.num_beams = num_beams
        self.temperature = temperature
        print('set num_beams: {}'.format(num_beams))
        print('set temperature: {}'.format(temperature))

    def upload(self, up_img=False, up_video=False, audio_flag=False):
        if up_img and not up_video:
            self.chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully, output only the letter code corresponding to the correct option, and explain your answers in detail."
            self.chat.upload_img(up_img, self.chat_state, self.img_list)
        elif not up_img and up_video:
            self.chat_state.system =  ""
            if audio_flag:
                self.chat.upload_video(up_video, self.chat_state, self.img_list)
            else:
                self.chat.upload_video_without_audio(up_video, self.chat_state, self.img_list)

    def ask_answer(self, user_message):
        self.chat.ask(user_message, self.chat_state)
        llm_message = self.chat.answer(conv=self.chat_state,
                                       img_list=self.img_list,
                                       num_beams=self.num_beams,
                                       temperature=self.temperature,
                                       max_new_tokens=300,
                                       max_length=1000)[0]

        return llm_message

    def reset(self):
        if self.chat_state is not None:
            self.chat_state.messages = list()
        if self.img_list is not None:
            self.img_list = list()
        self.set_para()

def preprocess_user_message(user_message):
    # 删除特殊符号和标点符号
    # user_message = re.sub(r'[^\w\s]', '', user_message)

    # 转换非ASCII字符（例如，将日语字符转换为ASCII字符）
    user_message = unidecode(user_message)

    return user_message

if __name__ == "__main__":
    dataset = MMBenchDataset(data_file='/mnt/xuyibo/Video-LLaMA/mmbench/mmbench_dev_en_20231003.tsv')
    args = parse_args()
    chatbot = ChatBot(args)

    # 打开一个 CSV 文件用于保存结果
    with open('inference_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['index', 'user_message', 'chatbot_response', 'correct_answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 遍历数据集进行推断
        for data_sample in dataset:
            # 获取用户消息
            if data_sample['context'] is not None:
                prompt = data_sample['context'] + ' ' + data_sample['question'] + ' ' + data_sample['options'] + '\n' + 'Output only the letter code corresponding to the most correct one (\'A\', \'B\', \'C\', \'D\', \'E\'):'
            else:
                prompt = data_sample['question'] + ' ' + data_sample['options'] + '\n' + 'Output only the letter code corresponding to the most correct one (\'A\', \'B\', \'C\', \'D\', \'E\'):'
            user_message = prompt
            
            user_message = preprocess_user_message(prompt)

            # 上传图像或视频
            chatbot.upload(up_img=data_sample['img']) 

            # 执行推断
            chatbot_response = chatbot.ask_answer(user_message=user_message)

            # 获取正确答案
            correct_answer = data_sample.get('answer', '')  # 如果没有答案字段，设为默认值 ''

            # 保存结果到 CSV 文件
            writer.writerow({
                'index': data_sample['index'],
                'user_message': user_message,
                'chatbot_response': chatbot_response,
                'correct_answer': correct_answer
            })

            chatbot.reset()

    print("Inference and result saving completed.")

    # while True:
    #     try:
    #         file_path = input('Input file path: ')
    #     except:
    #         print('Input error, try again.')
    #         continue
    #     else:
    #         if file_path == 'exit':
    #             print('Goodbye!')
    #             break
    #         if not os.path.exists(file_path):
    #             print('{} not exist, try again.'.format(file_path))
    #             continue

    #     # chatbot.upload(up_img=file_path)
    #     chatbot.upload(up_video=file_path, audio_flag=True)

    #     while True:
    #         try:
    #             user_message = input('User: ')
    #         except:
    #             print('Input error, try again.')
    #             continue
    #         else:
    #             if user_message == 'para':
    #                 num_beams = int(input('Input new num_beams:(1-10) '))
    #                 temperature = float(input('Input new temperature:(0.1-2.0) '))
    #                 chatbot.set_para(num_beams=num_beams, temperature=temperature)
    #                 continue
    #             if user_message == 'exit':
    #                 break
            
    #         llm_message = chatbot.ask_answer(user_message=user_message)
    #         print('ChatBot: {}'.format(llm_message))

    #     chatbot.reset()