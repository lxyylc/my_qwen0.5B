from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# 这个名字要和huggingface上实际存在的模型文件名相同，要不然就会报错
MODEL_NAME = "Qwen/Qwen1.5-0.5B"
LOCAL_MODEL_DIR = os.path.join(os.getcwd(), "qwen0_5B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def load_qwen_model():
    config_path = os.path.join(LOCAL_MODEL_DIR, MODEL_NAME, "config.json")
    if not os.path.exists(config_path):
        print(f"本地未找到Qwen-0.5B模型，开始自动下载到：{LOCAL_MODEL_DIR}")
    else:
        print(f"本地已存在Qwen-0.5B模型，直接加载：{config_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=LOCAL_MODEL_DIR
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        cache_dir=LOCAL_MODEL_DIR,
    )
    model.eval()  # 推理模式
    return tokenizer, model



def generate_text(tokenizer, model, prompt, max_new_tokens=200, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_only = full_text[len(prompt):]
    return full_text, generated_only



if __name__ == "__main__":

    tokenizer, model = load_qwen_model()
    test_prompt = "大模型剪枝是一种模型压缩技术，其核心原理是"
    print(f"\n输入文本：{test_prompt}")
    print("=" * 50 + "\n生成结果：")

    full_text, generated_only = generate_text(tokenizer, model, test_prompt)
    print(full_text)
    print(f"\n模型文件已保存到：{LOCAL_MODEL_DIR}")