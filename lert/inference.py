from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

model_path = '/mnt/cfs/NLP/zcl/subjects_classification/lert/output_badclean_pretrain_rl_hard_512/checkpoint-48000'


model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=10)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, max_length=256, padding="max_length", 
                                          truncation=True,padding_side='right', truncation_side='left')

classifier = pipeline('text-classification', 
                      model=model, 
                      tokenizer=tokenizer,
                      max_length=256,
                      padding="max_length",
                      truncation=True,
                      use_fast=True,
                      )

text = '3. 下面是用一套七巧板拼成的鸟,你知道是怎样拼出来的吗?可以在图上分-分,也可以将画好或拼好的作品贴在这里'

# print(tokenizer(text, padding="max_length", max_length=256, truncation=True))

result = classifier(text)
print(result)