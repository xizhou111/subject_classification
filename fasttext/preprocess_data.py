from collections import Counter
import json
from tqdm import tqdm
import jieba
from transformers import AutoTokenizer

def convert_to_fasttext_format(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        data = json.load(f_in)
        for item in tqdm(data, desc=f'Converting {input_file}'):
            question = item['question']
            if item['subject_id'] == 3:
                question_cut = question
            else:
                question_cut = ' '.join(jieba.cut(question)) # 使用jieba分词    
            f_out.write(f'__label__{item["subject_id"]} {question_cut}\n')


from multiprocessing import Pool
from functools import partial
tokenizer = AutoTokenizer.from_pretrained('/mnt/cfs/NLP/zcl/huggingface/models/chinese-roberta-wwm-ext')
def tokenize_and_format(item):
    question = item['question']
    question_cut = ' '.join(tokenizer.tokenize(question))
    return f'__label__{item["subject_id"]} {question_cut}\n'
def convert_to_fasttext_format_by_tokenizer(input_file, output_file):
    with open(input_file, 'r') as f_in:
        data = json.load(f_in)

    with Pool(128) as p:
        results = list(tqdm(p.imap(tokenize_and_format, data), total=len(data), desc=f'Converting fasttext'))

    with open(output_file, 'w') as f_out:
        f_out.writelines(results)


def remove_less_than_10000(input_file, output_file):
    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()
        labels = [line.split(' ')[0] for line in lines]
        label_counter = Counter(labels)

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            label = line.split(' ')[0]
            if label_counter[label] >= 10000:
                f_out.write(line)

if __name__ == '__main__':

    # 对训练集、验证集和测试集进行转换
    # convert_to_fasttext_format_by_tokenizer('/mnt/cfs/NLP/zcl/subjects_classification/fasttext/datasets/train_data_reduce_chinese.json', 
    #                                         'datasets/train_data_reduce_chinese_tokenizer.txt')
    # convert_to_fasttext_format_by_tokenizer('/mnt/cfs/NLP/zcl/subjects_classification/datasets/valid_data.json', 'datasets/valid_data.txt')
    # convert_to_fasttext_format_by_tokenizer('/mnt/cfs/NLP/zcl/subjects_classification/datasets/test_data.json', 'datasets/test_data.txt')
    convert_to_fasttext_format_by_tokenizer('/mnt/cfs/NLP/zcl/subjects_classification/fasttext/eval_data/eval_data.json', 
                                            '/mnt/cfs/NLP/zcl/subjects_classification/fasttext/eval_data/eval_data_tokenizer.txt')


    # # 对训练集、验证集和测试集进行处理
    # remove_less_than_10000('datasets/train_data.txt', 'datasets/train_data_filtered.txt')
    # remove_less_than_10000('datasets/valid_data.txt', 'datasets/valid_data_filtered.txt')
    # remove_less_than_10000('datasets/test_data.txt', 'datasets/test_data_filtered.txt')

    # # 删除原始数据
    # import os
    # os.remove('datasets/train_data.txt')
    # os.remove('datasets/valid_data.txt')
    # os.remove('datasets/test_data.txt')

    # # 修改过滤后的数据名称
    # os.rename('datasets/train_data_filtered.txt', 'datasets/train_data.txt')
    # os.rename('datasets/valid_data_filtered.txt', 'datasets/valid_data.txt')
    # os.rename('datasets/test_data_filtered.txt', 'datasets/test_data.txt')

    # # 统计一下转换后的每个学科的数据量
    # subject_ids = []
    # with open('datasets/train_data.txt', 'r') as f:
    #     for line in f:
    #         subject_ids.append(line.strip().split(' ')[0].replace('__label__', ' '))
    # subject_id_counter = Counter(subject_ids)
    # for subject_id, count in subject_id_counter.items():
    #     print(f"Train Subject ID: {subject_id}, Count: {count}")

    # subject_ids = []
    # with open('datasets/valid_data.txt', 'r') as f:
    #     for line in f:
    #         subject_ids.append(line.strip().split(' ')[0].replace('__label__', ' '))
    # subject_id_counter = Counter(subject_ids)
    # for subject_id, count in subject_id_counter.items():
    #     print(f"Valid Subject ID: {subject_id}, Count: {count}")

    # subject_ids = []
    # with open('datasets/test_data.txt', 'r') as f:
    #     for line in f:
    #         subject_ids.append(line.strip().split(' ')[0].replace('__label__', ' '))
    # subject_id_counter = Counter(subject_ids)
    # for subject_id, count in subject_id_counter.items():
    #     print(f"Test Subject ID: {subject_id}, Count: {count}")