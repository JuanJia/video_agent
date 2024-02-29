'''
找到complex.json和ErrorsCases_VideoLLaMa.json中video_path的交集，构建新的JSON数据
'''

import json
# 读取ErrorsCases_VideoLLaMa.json文件
with open('/mnt/xuyibo/Video-LLaMA/ErrorsCases_VideoLLaMa.json', 'r') as error_cases_file:
    error_cases_data = json.load(error_cases_file)
    error_examples = error_cases_data.get('error_examples', [])

# 读取之前生成的output.json文件
with open('/mnt/xuyibo/Video-LLaMA/complex.json', 'r') as output_file:
    output_data = json.load(output_file)

# 找到两个JSON文件中video_path的交集
common_video_paths = set([item['video_path'] for item in output_data]) & set([item['video'] for item in error_examples])

# 构建新的JSON数据
common_examples = []
for item in error_examples:
    if item['video'] in common_video_paths:
        common_examples.append({
            'task_type': item['task_type'],
            'video': item['video'],
            'question': item['question'],
            'pred': item['pred'],
            'gt': item['gt']
        })

# 构建新的JSON格式
result_json = {
    'complex_error_examples': common_examples
}

# 将结果保存为新的JSON文件
with open('/mnt/xuyibo/Video-LLaMA/complex_error_examples.json', 'w') as common_file:
    json.dump(result_json, common_file, indent=2)
