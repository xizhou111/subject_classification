import json
import os
import pandas as pd
from pprint import pprint
import requests
import uuid
import jieba
from tqdm import tqdm

import fasttext

from preprocess_data import convert_to_fasttext_format
from train import get_all_predictions, calculate_metrics


def process_data(id_question):
    data ={}
    data['question'] = id_question['question']
    data['trace_id'] = str(uuid.uuid1())

    # res = requests.post('http://127.0.0.1:9007/cut_question',json=data)    
    res = requests.post('http://xbtk-internal.100tal.com/cut-question/cut_question',json=data)
    
    # pprint.pprint(json.loads(res.text))
    res = json.loads(res.text)
    return res['data']['question_clean']

if __name__ == '__main__':

    subject_map_file = '/mnt/cfs/NLP/zcl/subjects_classification/category_map.json'
    # 读取学科映射
    with open(subject_map_file, 'r') as f:
        subject_map = json.load(f)

    # 如果eval_data.txt已经存在，直接加载模型进行预测
    if not os.path.exists('eval_data/eval_data_tokenizer.txt'):

        eval_data_file = '/mnt/cfs/NLP/zcl/subjects_classification/datasets/学科学段分类器—20240409.xlsx'

        # 读取数据
        eval_data = pd.read_excel(eval_data_file)

        data = []
        for index, row in eval_data.iterrows():
            data.append({
                'question': row['result_texts'],
                'subject_id': subject_map[row['学科']]
            })

        results = []
        for item in data:
            # res = process_data(item)
            data_cleaned = {
                "subject_id": item['subject_id'],
                "question": process_data(item),
            }
            # pprint(data_cleaned)
            results.append(data_cleaned)

        # 保存数据
        with open('eval_data/eval_data.json', 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        # 转换数据格式
        convert_to_fasttext_format('eval_data/eval_data.json', 'eval_data/eval_data.txt')
    
    # 加载模型
    classifier = fasttext.load_model('models/subject_cls_fasttext_model_tokenizer.bin')

    # 获取预测结果
    data_files = ['eval_data/eval_data_tokenizer.txt']
    predictions = get_all_predictions(classifier, data_files)

    eval_labels, eval_predicted = predictions[0]

    # 计算并打印指标
    eval_accuracy, eval_precision, eval_recall, eval_f1, classification_rep = calculate_metrics(eval_labels, eval_predicted)

    pprint(f"Eval data: Accuracy: {eval_accuracy:.4f}, Precision: {eval_precision:.4f}, Recall: {eval_recall:.4f}, F1-score: {eval_f1:.4f}, \nClassification Report: \n{classification_rep}")

    id2label = {v: k for k, v in subject_map.items()}
    # 利用classficaiton_report中的recall值，计算每个类别的正确率
    for label, value in classification_rep.items():
        try:
            # 尝试将标签转换为整数
            int_label = int(label)
        except ValueError:
            # 如果转换失败，跳过这个标签
            continue
        if int_label in id2label:
            print(f"{id2label[int_label]}: {value['recall']:.4f}")



    

