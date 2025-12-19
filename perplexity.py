import os
import torch
import torch.nn.functional as F
import warnings
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from qwen0_5B import load_qwen_model
MODEL_DIR = os.path.join(os.getcwd(), "qwen0_5B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_SEQ_LEN = 512
BATCH_SIZE = 4

def load_wikitext2(tokenizer):
    """加载并预处理WikiText-2数据集"""
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    print("预处理WikiText-2数据...")

    # 过滤空行
    def filter_empty(example):
        text = example["text"].strip()
        return len(text) > 0 and not text.startswith("=")

    dataset = dataset.filter(filter_empty)

    # 拼接所有文本
    all_text = "\n".join(dataset["text"])

    # 按最大长度分块（更合理的方式）
    tokenized = tokenizer(
        all_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False
    )

    tokens = tokenized["input_ids"][0]

    # 按MAX_SEQ_LEN分块，步长为MAX_SEQ_LEN（无重叠）
    chunks = []
    for i in range(0, len(tokens), MAX_SEQ_LEN):
        chunk = tokens[i:i + MAX_SEQ_LEN]
        if len(chunk) >= 64:  # 过滤太短的块
            chunks.append(chunk)

    # 分批处理
    batches = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]

        # 对齐批次内序列长度
        max_len_in_batch = min(max(len(c) for c in batch_chunks), MAX_SEQ_LEN)

        batch_tensors = []
        for chunk in batch_chunks:
            if len(chunk) > max_len_in_batch:
                chunk = chunk[:max_len_in_batch]
            elif len(chunk) < max_len_in_batch:
                # 填充到batch最大长度,好像这个elif下的语句根本没有被执行，生成的token根本不需填充padding
                padding = torch.full((max_len_in_batch - len(chunk),),
                                     tokenizer.pad_token_id,
                                     dtype=torch.long)
                chunk = torch.cat([chunk, padding])
            batch_tensors.append(chunk)

        batch = torch.stack(batch_tensors).to(DEVICE)
        batches.append(batch)

    print(f"数据集预处理完成：共 {len(batches)} 批次，总token数 {sum(len(c) for c in chunks)}")
    return batches


def calculate_perplexity(model, tokenizer, batches):
    """计算困惑度（修正版）"""
    print("开始计算PPL...")
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(batches, desc="计算PPL进度"):
            # 创建attention_mask（忽略padding）
            attention_mask = (batch != tokenizer.pad_token_id).long()

            # 获取模型输出
            outputs = model(batch, attention_mask=attention_mask, output_hidden_states=False)
            logits = outputs.logits

            # 计算每个位置的损失
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            # 计算交叉熵损失
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # 只计算非padding位置的损失
            losses = losses.view(shift_labels.size())
            valid_losses = losses * shift_mask

            # 累加
            valid_tokens = shift_mask.sum().item()
            if valid_tokens > 0:
                total_nll += valid_losses.sum().item()
                total_tokens += valid_tokens

    if total_tokens == 0:
        return float('inf')

    avg_nll = total_nll / total_tokens
    ppl = np.exp(avg_nll)
    return ppl


if __name__ == "__main__":
    tokenizer, model = load_qwen_model()
    batches = load_wikitext2(tokenizer)

    # 测试一个批次看看
    if len(batches) > 0:
        sample_batch = batches[0]
        print(f"批次形状: {sample_batch.shape}")
        print(f"非padding比例: {(sample_batch != tokenizer.pad_token_id).float().mean():.2%}")

    ppl = calculate_perplexity(model, tokenizer, batches)
    print(f"\nQwen-0.5B 在WikiText-2测试集上的困惑度（PPL）：{ppl:.2f}")