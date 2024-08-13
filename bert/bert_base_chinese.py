import json
import os
import pandas as pd
import requests
from pprint import pprint
# import logging

import datasets
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

# import evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"  # 选择GPU




if __name__ == '__main__':
    if not os.path.exists('processed_data'):
        # train_data.json, valid_data.json, test_data.json
        data_file = '../datasets'

        # 读取数据, 文本是“question”，标签是“subject_id”, “question”换成“text”，“subject_id”换成“label”
        train_data = pd.read_json(os.path.join(data_file, 'train_data.json'))
        val_data = pd.read_json(os.path.join(data_file, 'valid_data.json'))
        test_data = pd.read_json(os.path.join(data_file, 'test_data.json'))

        # 更换字段名，并移除无用字段
        train_data = train_data.rename(columns={"question": "text", "subject_id": "label"})
        val_data = val_data.rename(columns={"question": "text", "subject_id": "label"})
        test_data = test_data.rename(columns={"question": "text", "subject_id": "label"})
        train_data = train_data.drop(columns=["question_cut"])
        val_data = val_data.drop(columns=["question_cut"])
        test_data = test_data.drop(columns=["question_cut"])

        # 将DataFrame转换为Dataset
        train_dataset = Dataset.from_pandas(train_data)
        val_dataset = Dataset.from_pandas(val_data)
        test_dataset = Dataset.from_pandas(test_data)

        # 将数据集放入一个字典中
        data = DatasetDict({
            "train": train_dataset,
            "valid": val_dataset,
            "test": test_dataset
        })

    model_path = "/mnt/cfs/NLP/zcl/huggingface/models/bert-base-chinese"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")
    
    processed_data_dir = "/mnt/cfs/NLP/zcl/subjects_classification/bert/processed_data"
    if os.path.exists(processed_data_dir):
        tokenized_datasets = DatasetDict.load_from_disk(processed_data_dir)
    else:
        tokenized_datasets = data.map(tokenize_function, batched=True)
        tokenized_datasets.save_to_disk(processed_data_dir)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=256)
    
    # 定义评估指标，包括准确率、精确率、召回率、F1值
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    label2id = {"其他": 0, "语文": 1, "数学": 2, "英语": 3, "物理": 4, "化学": 5, "生物": 6, "历史": 7, "地理": 8, "政治": 9}
    id2label = {0: "其他", 1: "语文", 2: "数学", 3: "英语", 4: "物理", 5: "化学", 6: "生物", 7: "历史", 8: "地理", 9: "政治"}
    print("load model")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, 
                                                               num_labels=10,
                                                               id2label=id2label,
                                                               label2id=label2id
                                                               )   

    training_args = TrainingArguments(
        output_dir="./bert_base_chinese_output",
        overwrite_output_dir=True,
        learning_rate=2e-5,
        per_device_train_batch_size=330,
        per_device_eval_batch_size=330,
        num_train_epochs=1,
        eval_steps=200,
        evaluation_strategy="steps",
        metric_for_best_model="accuracy",
        save_steps=200, 
        save_strategy="steps",
        save_total_limit=5,
        save_safetensors=False,
        load_best_model_at_end=True,
        weight_decay=0.01,
        logging_dir="./bert_base_chinese_logs",
        logging_first_step=True,
        # log_level=20,
        logging_strategy="steps",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 设置日志级别
    # logging.set_verbose_warning()

    print("Start training")

    train_result = trainer.train()

    # 记录训练日志
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    # 记录验证日志
    eval_result = trainer.evaluate()
    trainer.log_metrics("eval", eval_result)
    trainer.save_metrics("eval", eval_result)



