import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from qwen0_5B import load_qwen_model
# 超参数定义
MAX_SEQ_LEN = 2048
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_test_data(name, tokenizer, seq_len=2048, batch_size=4):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)

    def process_data(samples, tokenizer, seq_len, field_name):
        # 拼接所有文本并tokenize
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        # 按序列长度切分数据
        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    if 'wikitext2' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=2048, batch_size=32, device="cuda"):
    model.to(device)
    model.eval()
    ppls = {}

    for dataset in datasets:
        test_loader = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size)
        nlls = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {dataset}"):
                batch = batch.to(device)

                output = model(batch, use_cache=False)
                lm_logits = output.logits

                # 检查logits是否有异常值
                if torch.isfinite(lm_logits).all():
                    # 位移计算损失（预测下一个token）
                    shift_logits = lm_logits[:, :-1, :].contiguous()
                    shift_labels = batch[:, 1:].contiguous()

                    # 计算交叉熵损失
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    nlls.append(loss)

        # 计算困惑度
        if nlls:
            ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
            ppls[dataset] = ppl
        else:
            ppls[dataset] = float('inf')

    print("PPL after evaluation: {}".format(ppls))
    return ppls



if __name__ == "__main__":
    print("Loading Qwen-0.5B model...")
    tokenizer, model = load_qwen_model()

    print("Starting PPL evaluation on wikitext2...")
    ppl_results = ppl_eval(
        model=model,
        tokenizer=tokenizer,
        datasets=['wikitext2'],  # 只测试wikitext2
        model_seq_len=2048,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    print(f"Final PPL result: {ppl_results}")
