# 中国語ネットスラングの標準語変換（T5 / mT5）

SNS などで使われる中国語のネットスラングや不自然な表現を、自然な標準語に書き換える Seq2Seq モデルです。
同じタスクを BART で解いた [NLP-bart-slang-normalization](https://github.com/waimi0228/NLP-bart-slang-normalization) と同じデータセットを使い、T5 系モデルで実装しました。

| 入力 | 出力（目標） |
|---|---|
| 你這樣真的很母湯 | 你這樣真的很不行 |
| 你踹共 | 你出來說 |

## 手法
- **mt5.py（最終版）**
  - 事前学習済みモデル `google/mt5-small` を fine-tuning
  - URL・記号の除去、句読点の統一などの前処理、`correct:` プレフィックス付き入力
  - train : test = 8 : 2、評価指標 ROUGE、20 epochs、learning rate 3e-5
  - GPU メモリ不足に対応するため CUDA allocator の設定を調整
- **mt5_v1.py**：mT5 の初期版（10 epochs）
- **t5_small.py**：英語版 `t5-small` によるベースライン（`normalize:` プレフィックス付き）

## データ
`data/data0519.csv`：誤った表現 → 修正後の表現 のペア（約 900 件、自作）

## 使用技術
Python / PyTorch / Hugging Face Transformers / datasets / evaluate

## 実行方法
```bash
pip install -r requirements.txt
python mt5.py
```

学習済みモデル（約 1GB）はサイズの都合でリポジトリに含めていません。実行すると `my_mt5_model/` に保存されます。
