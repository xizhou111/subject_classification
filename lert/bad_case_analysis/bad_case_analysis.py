import os
from transformers import (pipeline,
                          AutoModelForSequenceClassification,
                          AutoTokenizer
                          )
import time
import json
from tqdm import tqdm
from pprint import pprint

from sklearn.metrics import (accuracy_score, 
                             precision_recall_fscore_support, 
                             classification_report,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 选择GPU


# 定义评估指标，包括准确率、精确率、召回率、F1值
def compute_metrics(labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    classification_rep = classification_report(labels, predictions, output_dict=True, digits=4)
    confusion = confusion_matrix(labels, predictions,labels=[0,1,2,3,4,5,6,7,8,9])
    results = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    #添加每个类别的recall指标到结果字典
    for label, metrics in classification_rep.items():
        if isinstance(metrics, dict):
            # results[f'recall_{label}'] = metrics['recall']
            try:
                results[f'recall_{id2label[int(label)]}'] = metrics['recall']
            except:
                results['label'] = metrics['recall']

    # 绘制混淆矩阵并保存
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('./confusion_matrix.png')

    return results

if __name__ == '__main__':

    checkpoint = '/mnt/cfs/NLP/zcl/subjects_classification/lert/output_badclean_hard_512/checkpoint-74000'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, padding_side='right', truncation_side='left')
    classifier = pipeline('text-classification', 
                          model=checkpoint, 
                          tokenizer=tokenizer, 
                          truncation=True, 
                          use_fast=True,
                          max_length=256, 
                          padding="max_length",
                          device=1,
                          )

    eval_data_file = '/mnt/cfs/NLP/zcl/subjects_classification/datasets/eval_clean_data.json'
    with open(eval_data_file, 'r') as f:
        eval_data = json.load(f)

    id2label = {0: "其他", 1: "语文", 2: "数学", 3: "英语", 4: "物理", 5: "化学", 6: "生物", 7: "历史", 8: "地理", 9: "政治"}

    labels = []
    predictions = []
    with open('./prediction_errors.txt', 'w') as error_file:  # 创建一个新的文件来保存预测错误的结果
        for eval_batch in tqdm(eval_data):
            sentence = eval_batch['question']
            result = classifier(sentence)
            labels.append(int(eval_batch['subject_id']))
            predictions.append(int(result[0]['label']))

            if labels[-1] != predictions[-1]:
                error_file.write(f"question: {sentence}\n")  # 将问题写入文件
                error_file.write(f"label: {id2label[labels[-1]]}, prediction: {id2label[predictions[-1]]}, score: {result[0]['score']}\n")  # 将标签和预测写入文件
                error_file.write("\n")  # 添加一个空行

    results = compute_metrics(labels, predictions)
    pprint(results)