import os 
import json
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
from threading import Lock

from inference import Classifier

lock = Lock()

def process_file(json_file):
    classifier = Classifier()
    difficult_samples = []
    json_fold = '/mnt/cfs/NLP/wx/data/tusou/xpp_sql_data/jsons'

    if not json_file.endswith('.json'):
        return []

    with open(os.path.join(json_fold, json_file), 'r') as f:
        data = json.load(f)
        for item in tqdm(data, total=len(data)):
            question = item['question_txt']
            if len(question) < 20:
                continue
            label, score, question_clean = classifier.predict(question)
            if label != item['subject_id'] or score < 0.7:
                difficult_samples.append({
                    "question_clean": question_clean,
                    "question_cut": question_clean,
                    "subject_id": item['subject_id'],
                    "predict_label": label,
                })

    if len(difficult_samples) == 0:
        return []

    with lock:
        with open('/mnt/cfs/NLP/zcl/subjects_classification/datasets/difficult_samples/{}'.format(json_file), 'w') as f:
            json.dump(difficult_samples, f, ensure_ascii=False, indent=2)

    return difficult_samples

if __name__ == '__main__':
    # set_start_method('spawn', force=True)
    json_fold = '/mnt/cfs/NLP/wx/data/tusou/xpp_sql_data/jsons'
    json_files = os.listdir(json_fold)

    with Pool(8) as p:
        results = p.map(process_file, json_files)

    results = [item for sublist in results for item in sublist]

    print(f"difficult samples: {len(results)}")

    json.dump(results, open('/mnt/cfs/NLP/zcl/subjects_classification/datasets/difficult_samples.json', 'w'), 
                        ensure_ascii=False, indent=2)