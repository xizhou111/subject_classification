import os
import pandas as pd
import json
from pprint import pprint

from datasets import Dataset

from sklearn.metrics import (accuracy_score, 
                             precision_recall_fscore_support, 
                             classification_report,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          DataCollatorWithPadding, 
                          Trainer, 
                          TrainingArguments,
                          EarlyStoppingCallback,
                          )
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # 选择GPU
os.environ["TOKENIZERS_PARALLELISM"] = "True"  # 加速tokenizer

if __name__ == '__main__':

    # Load the evaluation dataset
    eval_data_file = '/mnt/cfs/NLP/zcl/subjects_classification/datasets/eval_clean_data.json'
    eval_data = pd.read_json(eval_data_file)

    eval_data = eval_data.rename(columns={"question": "text", "subject_id": "label"})
    # text的所有字符串后面都加上 "APP扫码看讲解"
    # eval_data['text'] = eval_data['text'].apply(lambda x: x + "APP扫码看讲解")
    eval_dataset = Dataset.from_pandas(eval_data)


    # output_512/checkpoint-32000
    # output_pretrain_512/checkpoint-30000
    # output_pretrain_sampler_512/checkpoint-32000
    # output_badclean_pretrain_sampler_512/checkpoint-32000
    # output_badclean_pretrain_512/checkpoint-28000
    model_path = '/mnt/cfs/NLP/zcl/subjects_classification/roberta/output_badclean_pretrain_deduplication_rdrop_512/checkpoint-60000'
    model_path = '/mnt/cfs/NLP/zcl/subjects_classification/roberta/output_badclean_pretrain_sampler_512/checkpoint-32000'
    model_path = '/mnt/cfs/NLP/zcl/subjects_classification/roberta/output_badclean_pretrain_512/checkpoint-28000'
    model_path = '/mnt/cfs/NLP/zcl/subjects_classification/roberta/output_badclean_pretrain_512/checkpoint-28000'
    model_path = '/mnt/cfs/NLP/zcl/subjects_classification/roberta/output_pretrain_512/checkpoint-30000'


    # Load the model
    print("load model:", model_path)
    label2id = {"其他": 0, "语文": 1, "数学": 2, "英语": 3, "物理": 4, "化学": 5, "生物": 6, "历史": 7, "地理": 8, "政治": 9}
    id2label = {0: "其他", 1: "语文", 2: "数学", 3: "英语", 4: "物理", 5: "化学", 6: "生物", 7: "历史", 8: "地理", 9: "政治"}
    model = AutoModelForSequenceClassification.from_pretrained(model_path, 
                                                               num_labels=10,
                                                               id2label=id2label,
                                                               label2id=label2id
                                                               )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length", return_tensors="pt")
    
    tokenized_datasets = eval_dataset.map(tokenize_function, batched=True)

    # 定义评估指标，包括准确率、精确率、召回率、F1值
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
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

        # 添加每个类别的recall指标到结果字典
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
        plt.savefig('./eval_output_pretrain_sampler_512/confusion_matrix.png')

        return results

    train_args = TrainingArguments(
        output_dir='./eval_output_pretrain_sampler_512',
        logging_dir='./eval_output_pretrain_sampler_512',
        per_device_eval_batch_size=256,
    )

    trainer = Trainer(
        args=train_args,
        model=model,
        eval_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    eval_result = trainer.evaluate()

    trainer.log_metrics("eval", eval_result)
    trainer.save_metrics("eval", eval_result)

    pprint(eval_result)



