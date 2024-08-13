import fasttext
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

# 获取模型的预测结果
def get_all_predictions(classifier, data_files):
    results = []
    for data_file in data_files:
        labels, texts = [], []
        with open(data_file, 'r') as f:
            for line in tqdm(f, desc=f'Processing {data_file}'):
                split_line = line.strip().split(' ')
                labels.append(split_line[0].replace('__label__', ''))
                texts.append(' '.join(split_line[1:]))
        # 关闭文件
        f.close()
        predicted_labels = [
                label[0].replace('__label__', '') for label in tqdm(classifier.predict(texts)[0])
            ]
        results.append((labels, predicted_labels))
    return results

# 计算指标
def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    accuracy = accuracy_score(true_labels, predicted_labels)
    classification_rep = classification_report(true_labels, predicted_labels, digits=4, output_dict=True)
    return accuracy, precision, recall, f1, classification_rep

# 计算每个类别的正确率
def calculate_class_correct_proportions(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    class_correct_proportions = np.diag(cm) / np.sum(cm, axis=1)
    return class_correct_proportions

if __name__ == '__main__':
    # 训练模型
    classifier = fasttext.train_supervised(input='datasets/train_data_reduce_chinese_tokenizer.txt',
                                        lr = 1.0,
                                        wordNgrams = 2,
                                        epoch = 25,
                                        # pretrainedVectors = 'cc.zh.300.vec',
                                        dim = 300,
                                        )
    # classifier = fasttext.train_supervised( input='datasets/train_data.txt',
    #                                         autotuneValidationFile='datasets/valid_data.txt',
    #                                         # autotuneMetric='f1:__label__7',
    #                                         autotuneDuration=36,
    #                                         )

    # # 获取预测结果
    # data_files = ['datasets/train_data.txt', 'datasets/valid_data.txt', 'datasets/test_data.txt']
    # predictions = get_all_predictions(classifier, data_files)

    # # 获取预测结果
    # train_labels, train_predicted = predictions[0]
    # valid_labels, valid_predicted = predictions[1]
    # test_labels, test_predicted = predictions[2]

    # # 计算并打印指标
    # train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(train_labels, train_predicted)
    # valid_accuracy, valid_precision, valid_recall, valid_f1 = calculate_metrics(valid_labels, valid_predicted)
    # test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(test_labels, test_predicted)

    # print(f"Train data: Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-score: {train_f1:.4f}")
    # print(f"Valid data: Accuracy: {valid_accuracy:.4f}, Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}, F1-score: {valid_f1:.4f}")
    # print(f"Test data: Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-score: {test_f1:.4f}")

    # # 计算每个类别的正确率
    # valid_class_correct_proportions = calculate_class_correct_proportions(valid_labels, valid_predicted)
    # test_class_correct_proportions = calculate_class_correct_proportions(test_labels, test_predicted)

    # # 打印每个类别的正确率
    # print("Valid data:")
    # for i, correct_proportion in enumerate(valid_class_correct_proportions):
    #     print(f"Class {i}: {correct_proportion:.4f}")
    # print("Test data:")
    # for i, correct_proportion in enumerate(test_class_correct_proportions):
    #     print(f"Class {i}: {correct_proportion:.4f}")

    # 保存模型
    classifier.save_model('models/subject_cls_fasttext_model_tokenizer.bin')
