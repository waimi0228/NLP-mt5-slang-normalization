import os
# 設定 CUDA allocator 減少碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
# 限制 CPU 端 OpenMP 執行緒
os.environ["OMP_NUM_THREADS"] = "16"

import torch
import pandas as pd
import re
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
import evaluate

# 1. 清理函數
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', str(text))
    text = re.sub(r'[^\w\s,，。！？]', '', text)
    text = re.sub(r'，', ',', text)
    text = re.sub(r'。', '.', text)
    return text.strip()

# 2. 載入並清理資料
df = pd.read_csv('data/data0519.csv', encoding='utf-8')
df = df.dropna(subset=["input_text", "target_text"]).copy()
df['cleaned_input_text'] = df['input_text'].apply(clean_text)
df['cleaned_target_text'] = df['target_text'].apply(clean_text)
df = df[(df['cleaned_input_text']!='') & (df['cleaned_target_text']!='')]

# 3. 切分資料集
train_df, valid_df = train_test_split(
    df[['cleaned_input_text','cleaned_target_text']],
    test_size=0.2,
    random_state=42
)
train_ds = Dataset.from_pandas(train_df).remove_columns(['__index_level_0__'])
valid_ds = Dataset.from_pandas(valid_df).remove_columns(['__index_level_0__'])

# 4. 載入模型與 tokenizer
model_name = 'google/mt5-small'
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)
model.gradient_checkpointing_enable()

# 5. Tokenize 函數
max_len = 64

def preprocess_function(batch):
    inputs = ['correct: ' + text for text in batch['cleaned_input_text']]
    model_inputs = tokenizer(
        inputs,
        max_length=max_len,
        padding='max_length',
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch['cleaned_target_text'],
            max_length=max_len,
            padding='max_length',
            truncation=True
        )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_train = train_ds.map(preprocess_function, batched=True)
tokenized_valid = valid_ds.map(preprocess_function, batched=True)

# 6. 評估函數 (ROUGE) - 分批 decode 避免 MemoryError
rouge = evaluate.load('rouge')
batch_decode_size = 8  # 減少一次 decode 的數量

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # 將 preds/labels 轉為純 Python list
    if hasattr(preds, 'tolist'):
        preds_list = preds.tolist()
    else:
        preds_list = list(preds)
    if hasattr(labels, 'tolist'):
        labels_list = labels.tolist()
    else:
        labels_list = list(labels)

    decoded_preds, decoded_labels = [], []
    # 分批進行 decode
    for i in range(0, len(preds_list), batch_decode_size):
        chunk_preds = preds_list[i:i+batch_decode_size]
        chunk_labels = labels_list[i:i+batch_decode_size]
        decoded_preds.extend(
            tokenizer.batch_decode(chunk_preds, skip_special_tokens=True)
        )
        decoded_labels.extend(
            tokenizer.batch_decode(chunk_labels, skip_special_tokens=True)
        )
    # 清理以釋放記憶體
    del preds_list, labels_list
    torch.cuda.empty_cache()

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    return {k: round(v, 4) for k, v in result.items()}

# 7. TrainingArguments & Trainer TrainingArguments & Trainer
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='no',  # 關閉訓練過程中的驗證以避免 OOM
save_strategy='no',        # 關閉自動存檔
# load_best_model_at_end 和 early stopping callback 只能在有 eval 時使用，故暫時移除
# load_best_model_at_end=True,
# metric_for_best_model='rougeL',
# greater_is_better=True,
per_device_train_batch_size=1,
per_device_eval_batch_size=1,
gradient_accumulation_steps=16,
fp16=True,
num_train_epochs=10,
logging_dir='./logs',
logging_steps=20,
save_total_limit=2,
report_to='none'

)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_valid,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
# )
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics

)

# 8. 訓練
torch.cuda.empty_cache()
trainer.train()

# 9. 儲存模型
tokenizer.save_pretrained('./my_mt5_model')
model.save_pretrained('./my_mt5_model')

# 10. 推論示範
def generate_text(input_text, max_length=128):
    prompt = 'correct: ' + input_text
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.cuda() for k, v in inputs.items()} if torch.cuda.is_available() else inputs
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    model.eval()
    samples = [
        '你這樣真的非常母湯',
        '這事她真的無敵生氣但是她還是忍住了',
    ]
    for s in samples:
        print(f"原句: {s}")
        print(f"修正: {generate_text(s)}\n")
