import os
import pandas as pd
import json
import requests
from pprint import pprint
# import logging

import datasets
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

import numpy as np

import torch

from transformers.utils import logging

import transformers
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          DataCollatorWithPadding, 
                          Trainer, 
                          TrainingArguments,
                          EarlyStoppingCallback,
                          )
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 选择GPU

if __name__ == '__main__':

    # Load the evaluation dataset
    eval_data_file = '/mnt/cfs/NLP/zcl/subjects_classification/fasttext/eval_data/eval_data.json'
    eval_data = pd.read_json(eval_data_file)

    eval_data = eval_data.rename(columns={"question": "text", "subject_id": "label"})
    eval_dataset = Dataset.from_pandas(eval_data)

    model_path = '/mnt/cfs/NLP/zcl/subjects_classification/bert/bert_base_chinese_output/checkpoint-4400'
    # Load the model
    label2id = {"其他": 0, "语文": 1, "数学": 2, "英语": 3, "物理": 4, "化学": 5, "生物": 6, "历史": 7, "地理": 8, "政治": 9}
    id2label = {0: "其他", 1: "语文", 2: "数学", 3: "英语", 4: "物理", 5: "化学", 6: "生物", 7: "历史", 8: "地理", 9: "政治"}
    print("load model")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, 
                                                               num_labels=10,
                                                               id2label=id2label,
                                                               label2id=label2id
                                                               )
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")
    
    tokenized_datasets = eval_dataset.map(tokenize_function, batched=True)


    # 定义评估指标，包括准确率、精确率、召回率、F1值
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
        acc = accuracy_score(labels, predictions)
        classification_rep = classification_report(labels, predictions, output_dict=True, digits=4)
        results = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
        # 添加每个类别的recall指标到结果字典
        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                results[f'recall_{label}'] = metrics['recall']
        return results

    trainer = Trainer(
        model=model,
        eval_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    eval_result = trainer.evaluate()

    trainer.log_metrics("eval", eval_result)
    trainer.save_metrics("eval", eval_result)

    pprint(eval_result)







