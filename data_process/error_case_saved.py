'''
通过正则匹配判断预测结果是否正确,将错误样例保存为json文件
'''

import re
import json

def get_first_option_in_parentheses(text):
    # 使用正则表达式匹配第一次出现的括号中的内容
    match = re.search(r'\((.*?)\)', text)
    return match.group(1) if match else None

def is_prediction_incorrect(pred, gt):
    # 使用正则表达式匹配括号中的内容，以便比较预测和真实值
    pred_option = get_first_option_in_parentheses(pred)
    gt_option = get_first_option_in_parentheses(gt)

    # 如果预测中的选项和真实值中的选项不一致，则认为预测错误
    return pred_option != gt_option

def filter_and_save_errors(input_json, output_json):
    with open(input_json, 'r') as file:
        data = json.load(file)

    error_examples = []

    for res in data['res_list']:
        if is_prediction_incorrect(res['pred'], res['gt']):
            error_example = {
                "task_type": res["task_type"],
                "video": res["video"],
                "question": res["question"],
                "pred": res["pred"],
                "gt": res["gt"]
            }
            error_examples.append(error_example)

            # 打印错误信息
            print(f"Error found for task_type: {res['task_type']}")
            print(f"Question: {res['question']}")
            print(f"Predicted: {res['pred']}")
            print(f"Ground truth: {res['gt']}")
            print("-" * 30)

    error_data = {"error_examples": error_examples}

    with open(output_json, 'w') as output_file:
        json.dump(error_data, output_file, indent=2)

if __name__ == '__main__':
    filter_and_save_errors('/mnt/xuyibo/Video-LLaMA/test.json', '/mnt/xuyibo/Video-LLaMA/ErrorsCases.json')
