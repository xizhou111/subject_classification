from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import requests
import json
import uuid
from pprint import pprint
import torch
from time import time

def process_data(text1):
    data ={}
    data['question'] = text1
    data['trace_id'] = str(uuid.uuid1())           

    # res = requests.post('http://127.0.0.1:9007/cut_question',json=data)    
    res = requests.post('http://xbtk-internal.100tal.com/cut-question/cut_question',json=data)
    
    # pprint(json.loads(res.text))
    return json.loads(res.text)['data']['question_clean']


class Classifier:
    def __init__(self, device_id=0):
        self.model_path = '/mnt/cfs/NLP/zcl/subjects_classification/roberta/output_pretrain_512/checkpoint-30000'
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=10)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, max_length=256, padding="max_length", truncation=True)
        self.device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.classifier = pipeline('text-classification', 
                                    model=self.model_path,
                                    tokenizer=self.model_path,
                                    max_length=256,
                                    truncation=True,
                                    use_fast=True,
                                    device=self.device
                                    )
        self.label2id = {"其他": 0, "语文": 1, "数学": 2, "英语": 3, "物理": 4, "化学": 5, "生物": 6, "历史": 7, "地理": 8, "政治": 9}

    def predict(self, text):
        text = process_data(text)
        result = self.classifier(text)
        label = self.label2id[result[0]['label']]
        score = result[0]['score']
        return label, score, text

    def predict_batch(self, texts):
        labels = []
        scores = []

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.nn.functional.softmax(logits, dim=-1)
            scores, indices = torch.max(predictions, 1)
            labels = [self.id2label[str(i)] for i in indices.tolist()]
        return labels, scores.tolist()
        
        # results = self.classifier(texts)

        # for result in results:
        #     label = self.id2label[result['label']]
        #     score = result['score']
        #     labels.append(label)
        #     scores.append(score)
        # return labels, scores


if __name__ == '__main__':
    texts = "$$  \\frac { 3 } { 5 } \\div \\left[ ( \\frac { 3 } { 4 } - \\frac { 2 } { 3 } ) \\div \\frac { 5 } { 6 } \\right] $$"
    classifier = Classifier()
    labels, scores = classifier.predict(texts)
    print(labels, scores)
