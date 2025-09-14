import torch
import ollama
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import re


# 通用封装函数
def generate_clean(model: str, prompt: str, temperature: float = 0.7, max_new_tokens: int = 256) -> str:
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
    )
    text = response["response"]

    # 如果是 deepseek 模型，去掉 <think> 内容
    if "deepseek" in model.lower():
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    return text.strip()


def init_model(args):
    model_path = args.model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 检查是否使用ollama
    if hasattr(args, 'use_ollama') and args.use_ollama:
        # 对于ollama，返回模型名称和设备
        return model_path, None, device
    else:
        # 传统方式加载模型
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer, device


def llama_reasoning(args, prompt, model, tokenizer, device, temperature=0.3, max_new_tokens=64, num_beams=1,
                    do_sample=True):
    # 检查是否使用ollama
    if tokenizer is None:
        try:
            return generate_clean(model, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        except Exception as e:
            print(f"Error in ollama reasoning: {e}")
            return ""
    else:
        # 使用传统模型
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_beams=num_beams,
            do_sample=do_sample,
        )
        output_ids = output_ids[0][len(inputs["input_ids"][0]):-1]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text


def llama_request1(args, prompt, model, tokenizer, device, temperature, max_new_tokens, do_sample=True):
    if tokenizer is None:
        try:
            generated_text = generate_clean(model, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            return generated_text
        except Exception as e:
            print(f"Error in ollama request1: {e}")
            return ""
    else:
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id
        )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output_text = output_text[len(prompt):]
        return output_text


def llama_request(model, tokenizer, input_prompt: list, time_interval, temperature=0.7):
    done = False
    resp = None

    if tokenizer is None:
        while not done:
            try:
                prompt = "".join(input_prompt)
                generated_text = generate_clean(model, prompt, temperature=temperature, max_new_tokens=256)
                if generated_text.startswith(input_prompt[0]):
                    generated_text = generated_text[len(input_prompt[0]):]

                resp = {
                    "choices": [
                        {"text": generated_text},
                    ]
                }
                done = True
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(time_interval)
    else:
        model.eval()
        while not done:
            try:
                input_ids = tokenizer(input_prompt, return_tensors="pt").to('cuda')
                output_ids = model.generate(**input_ids, max_new_tokens=256, temperature=temperature, do_sample=True,
                                            eos_token_id=tokenizer.eos_token_id)
                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

                resp = {
                    "choices": [
                        {"text": output_text[len(input_prompt[0]):]},
                    ]
                }
                done = True
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(time_interval)
    return resp
