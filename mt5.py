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
print(df)
# 3. 切分資料集
train_df, valid_df = train_test_split(
    df[['cleaned_input_text','cleaned_target_text']],
    test_size=0.2,
    random_state=42
)
train_ds = Dataset.from_pandas(train_df).remove_columns(['__index_level_0__'])
valid_ds = Dataset.from_pandas(valid_df).remove_columns(['__index_level_0__'])

print(len(train_ds), len(valid_ds))
# 4. 載入模型與 tokenizer
model_name = 'google/mt5-small'
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)
model.gradient_checkpointing_enable()

# 5. Tokenize 函數
max_len = 64
#
# def preprocess_function(batch):
#     inputs = ['correct: ' + text for text in batch['cleaned_input_text']]
#     model_inputs = tokenizer(
#         inputs,
#         max_length=max_len,
#         padding='max_length',
#         truncation=True
#     )
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(
#             batch['cleaned_target_text'],
#             max_length=max_len,
#             padding='max_length',
#             truncation=True
#         )
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

def preprocess_function(batch):
    inputs = ["correct: " + x for x in batch["cleaned_input_text"]]
    model_inputs = tokenizer(
        inputs,
        max_length=64,
        padding="max_length",
        truncation=True
    )

    # 先拿到原始的 target input_ids
    with tokenizer.as_target_tokenizer():
        raw_labels = tokenizer(
            batch["cleaned_target_text"],
            max_length=64,
            padding="max_length",
            truncation=True
        )["input_ids"]

    # 把 pad_token_id 全部換成 -100
    labels = [
        [ (tok if tok != tokenizer.pad_token_id else -100)
          for tok in seq ]
        for seq in raw_labels
    ]

    model_inputs["labels"] = labels
    return model_inputs

tokenized_train = train_ds.map(preprocess_function, batched=True)
tokenized_valid = valid_ds.map(preprocess_function, batched=True)

print(tokenized_train[0]["labels"])

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

    save_strategy='no',        # 關閉自動存檔
    # load_best_model_at_end 和 early stopping callback 只能在有 eval 時使用，故暫時移除
    # load_best_model_at_end=True,
    # metric_for_best_model='rougeL',
    # greater_is_better=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    # gradient_accumulation_steps=32,
    # fp16=True,
    learning_rate=3e-5,
    num_train_epochs=20,
    logging_dir='./logs',

    save_total_limit=2,
    report_to='none',

    logging_strategy="steps",
    logging_steps=10,           # 每 10 個 steps 印一次 loss
    # --- 也可以開啟 eval，印出 eval_loss ---
    evaluation_strategy="steps",
    eval_steps=100,             # 每 100 steps 做一次驗證

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

# 10. 推論函式示範
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 10. 推論函式示範（修正版）
def generate_text(input_text: str, max_new_tokens: int = 50) -> str:
    prompt = 'correct: ' + input_text
    # 1) Tokenize 並送到 GPU
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        padding='longest'
    ).to(device)

    # 2) 用 max_new_tokens & 明確指定 decoder_start_token_id
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        decoder_start_token_id=tokenizer.pad_token_id,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2,
    )

    # 3) decode 時跳過所有特殊符號
    #    skip_special_tokens=True 會自動去掉 <pad>、</s>、<extra_id_*>…
    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).strip()

    return text

if __name__ == '__main__':
    model.eval()
    samples = [
        '你這樣真的非常母湯',
        '你們踹共',
    ]
    for s in samples:
        print(f"原句: {s}")
        print(f"修正: {generate_text(s)}\n")


# # 10. 推論示範
# def generate_text(input_text: str, max_new_tokens: int = 50) -> str:
#     prompt = 'correct: ' + input_text
#     inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=50,
#         num_beams=5,
#         early_stopping=True,
#         decoder_start_token_id=tokenizer.pad_token_id,
#         no_repeat_ngram_size=2,
#         repetition_penalty=1.2,
#     )
#     # 保留哨兵 token 再手動剔除
#     pred = tokenizer.decode(outputs[0], skip_special_tokens=False)
#     return pred.replace('<extra_id_0>', '').strip()
# if __name__ == '__main__':
#     model.eval()
#     samples = [
#         '你這樣真的非常母湯',
#         '你們踹共',
#     ]
#     for s in samples:
#         print(f"原句: {s}")
#         print(f"修正: {generate_text(s)}\n")
