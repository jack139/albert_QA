## ALBERT阅读理解QA

### 测试数据集
CMRC 2018：篇章片段抽取型阅读理解（简体中文）

### 预训练权重
- google官方中文albert
- brightmart中文albert

### 运行环境
python 3.6
tensorflow 1.15

### 测试
```
python3 cmrc2018_finetune_albert.py --model albert_google --n_batch 16
python3 cmrc2018_finetune_albert.py --model albert_zh --n_batch 3
```

### 模型及相关代码来源
1. 官方Albert (https://github.com/google-research/albert)
2. brightmart预训练 (https://github.com/brightmart/albert_zh)
3. Bert中文finetune (https://github.com/ewrfcas/bert_cn_finetune)

