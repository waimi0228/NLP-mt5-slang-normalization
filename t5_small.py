import pandas as pd
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset


class TextNormDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.data = df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1) 加 prefix
        source = "normalize: " + self.data.loc[idx, 'input_text']
        target = self.data.loc[idx, 'target_text']

        # 2) source tokenize
        src_enc = self.tokenizer(
            source,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # 3) target tokenize
        tgt_enc = self.tokenizer(
            target,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        labels = tgt_enc['input_ids'].squeeze()
        # === 關鍵：把所有 pad token id 換成 -100，Trainer 才會 ignore index
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

        return {
            'input_ids': src_enc['input_ids'].squeeze(),
            'attention_mask': src_enc['attention_mask'].squeeze(),
            'labels': labels
        }


# 讀資料、初始化 tokenizer & dataset
df = pd.read_csv('data/data0519.csv', encoding='utf-8')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
dataset = TextNormDataset(df, tokenizer, max_len=128)

# 載 pre‑trained 模型
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# 訓練參數
args = TrainingArguments(
    output_dir='./model/',
    per_device_train_batch_size=8,
    num_train_epochs=5,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    save_total_limit=1,
    learning_rate=3e-5,
    warmup_steps=500,
    # step‑based logging 也可開啟看 loss:
    # logging_strategy="steps",
    # logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)
trainer.train()

# 儲存微調後模型
model.save_pretrained('./model/best_model')
tokenizer.save_pretrained('./model/best_model')

# === 推論示範 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
tok = tokenizer


def infer(src_text, max_new_tokens=50):
    prompt = "normalize: " + src_text
    enc = tok(prompt, return_tensors='pt', padding='longest', truncation=True).to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        decoder_start_token_id=tok.pad_token_id,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2
    )
    return tok.decode(out[0], skip_special_tokens=True).strip()


# 你的測試句
for s in [
    "你這樣真的非常母湯",
    "你們踹共",
    "這書真好讀到爆",
    "明天要去吃好吃的火鍋嗎？",
    "昨天我看了一部超級好笑電影"
]:
    print("原句:", s)
    print("修正:", infer(s), "\n")
