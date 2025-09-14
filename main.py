import argparse

from extraction.get_discrimination_prompt import get_discrimination_prompt
from extraction.get_final_prompt import get_final_qa_prompt
from llama_model import init_model, llama_reasoning, llama_request1
from extraction.Entities_Extraction import get_ner_sentence
from extraction.Explict_relation_extraction import get_explict_relation_prompt
from extraction.Implicit_relation_extraction import get_implicit_relation_prompt
from utils import load_data, load_prompt, extract_final_answer
import re
from uncertainty_selection import generate_uncertainty_qes

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction_temperature", type=float)
    parser.add_argument("--uncertainty_temperature", type=float)
    parser.add_argument('--dataset', help="dataset",
                        choices=["2wikimultihopQA", "GSM8K", "hotpotQA", "logiQA", "CommonsenseQA",
                                 "StrategyQA"])
    parser.add_argument("--model", choices=["llama2-7b", "llama2-13b", "llama3-8b"])
    parser.add_argument("--model_path", help="your local model path")
    parser.add_argument("--selfCon", default=False, type=bool, help="self consistency")
    parser.add_argument('--answer_extracting_prompt', default='The answer is', type=str)
    parser.add_argument('--type_list_file', default='./extraction/entity_type_list.txt', type=str)
    parser.add_argument("--infer_num", default='5', help='string')
    parser.add_argument("--ner_prompt", help='string')
    parser.add_argument("--seed",help="random seed")
    parser.add_argument("--method", type=str, choices=["zero_shot_cot", "few_shot_cot"], help="method")
    parser.add_argument("--inference_prompt_path", type=str, help="extraction path")
    parser.add_argument("--dataset_path", type=str, help="dataset path")
    parser.add_argument("--num_trials", type=int, default=1, help="number of trials to run for each question")
    parser.add_argument("--api_time_interval", type=int, default=1.0, help="api time interval")
    parser.add_argument("--use_ollama", action="store_true", help="Whether to use ollama model")
    args = parser.parse_args()

    args.extraction_temperature = 0.3  # temperature越低越严谨，越高越自由
    args.uncertainty_temperature = 0.7
    #args.model_path = "/root/autodl-tmp/SDA-CoT/llama-2-13b-chat-hf"
    args.dataset = "CommonsenseQA"
    #args.ner_prompt = '''Your task is to identify and extract entities from the sentence. Do NOT add any extra text, explanation, or punctuation! Follow these steps: 1. Read the input text carefully. 2. Find all entities based ONLY on the 'sentence'. 3. Outputting after 'entities:'''
    args.ner_prompt = "Extract all named entities from the given sentence. Only output the named entities as plain text, without explanations, labels, or additional information."
    args.method = 'few_shot_cot'

    if args.dataset == "2wikimultihopQA":
        args.dataset_path = "dataset/2wikimultihopQA/2wiki.json"
    elif args.dataset == "GSM8K":
        args.dataset_path = "dataset/gsm8k/gsm8k.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        args.inference_prompt_path = "./inference_prompts/gsm8k_k=10"
    elif args.dataset == "hotpotQA":
        args.dataset_path = "dataset/hotpotqa/hotpot.json"
    elif args.dataset == "logiQA":
        args.dataset_path = "dataset/logiqa/logiQA.json"
    elif args.dataset == "CommonsenseQA":
        args.dataset_path = "dataset/CommonsenseQA/CommonsenseQA.jsonl"
        args.inference_prompt_path = "./inference_prompts/csqa_k=10"
        args.direct_answer_trigger = "\nSo the answer is"
    elif args.dataset == "StrategyQA":
        args.dataset_path = "dataset/StrategyQA/StrategyQA.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
        args.inference_prompt_path = "./inference_prompts/strategyqa_k=10"




    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    
    # 在设置默认参数时，可以添加ollama相关设置
    args.use_ollama = True  # 如果想默认使用ollama，可以取消注释
    # 如果使用ollama，model_path应为ollama模型名称，如"llama2:7b"
    args.model_path = "deepseek-r1:32b"  # 取消注释并修改为您下载的模型名称
    
    return args

def main():
    args = parse_arguments()
    ner_prompt = args.ner_prompt
    questions, answers, ids = load_data(args)
    questions, answers, ids = questions[256:500], answers[256:500], ids[256:500]
    model, tokenizer, device = init_model(args)

    # 加载高质量 prompt
    prompt = load_prompt(args.inference_prompt_path)
    multipath = args.num_trials  # multipath 次数

    for idx, element in enumerate(questions):
        # 只保留CommonsenseQA answer choices之前的内容
        match = re.search(r'^(.*?)\nanswer choices:', element, re.DOTALL | re.IGNORECASE)
        question_only = match.group(1).strip() if match else element

        # 拼接高质量 prompt
        #full_prompt = f"{prompt}\nQ: {question_only}\nA:"

        all_answers = []
        for path_idx in range(multipath):
            # 实体抽取
            complete_ner_prompt = get_ner_sentence(ner_prompt, question_only)
            extraction_entities = llama_reasoning(args, complete_ner_prompt, model, tokenizer, device, temperature=0.1, max_new_tokens=64, num_beams=4, do_sample=True)
            # 显式关系抽取
            explicit_prompt = get_explict_relation_prompt(extraction_entities, question_only)
            extraction_explicit = llama_reasoning(args, explicit_prompt, model, tokenizer, device, temperature=0.3, max_new_tokens=32, num_beams=2, do_sample=True)
            # 隐式关系抽取
            implicit_prompt = get_implicit_relation_prompt(args, extraction_entities, extraction_explicit, question_only)
            extraction_implicit = llama_reasoning(args, implicit_prompt, model, tokenizer, device, temperature=0.4, max_new_tokens=64, num_beams=2, do_sample=True)

            #筛选有效隐式关系
            try:
                valid_relation = ""
                relation_inf_list = re.findall(r'[(](.*?)[)]', extraction_implicit)

                # 一次性获取所有关系的评分
                discrimination_prompt = get_discrimination_prompt(args, extraction_entities, extraction_implicit,
                                                                  element)
                scores_text = llama_reasoning(args, discrimination_prompt, model, tokenizer, device, temperature=0.3,
                                              max_new_tokens=256, num_beams=1, do_sample=True)

                # 逐个检查每个关系的评分
                for rk, item in enumerate(relation_inf_list):
                    # 提取关系的主要部分（去除可能的引号和空格）
                    item_parts = item.split(",")
                    if len(item_parts) >= 2:
                        # 获取关系的前两部分（通常是实体对）
                        entity_pair = ",".join(item_parts[:2]).strip()

                        # 构建更灵活的模式匹配多种格式
                        patterns = [
                            rf'\({entity_pair}.*?\).*?[-=].*?(\d+)',  # 匹配(1, 2, "xxx") - 8 或 (1, 2) = 8
                            rf'\({entity_pair}.*?\).*?(\d+)',  # 匹配其他可能的格式
                        ]

                        # 尝试所有模式
                        score_value = None
                        for pattern in patterns:
                            score_match = re.search(pattern, scores_text)
                            if score_match:
                                score_value = int(score_match.group(1))
                                break

                        # 如果找到评分且大于等于6，则添加到有效关系中
                        if score_value is not None and score_value >= 6:
                            item_str = "(" + item + ")"
                            valid_relation += item_str

                relation = extraction_explicit + valid_relation
            except Exception as e:
                print(f"Error in relationship validation: {e}")
                relation = extraction_explicit

            #拼接最终prompt（高质量prompt+实体+关系+问题）
            final_prompt = prompt + get_final_qa_prompt(args, extraction_entities, relation, element)
            # 用高质量 prompt 进行推理
            answer = llama_request1(args, final_prompt, model, tokenizer, device, temperature=0.7, max_new_tokens=256, do_sample=True)
            answer = extract_final_answer(args, answer)
            all_answers.append(answer)

        #对 all_answers 做投票、统计等处理
        print(f"Question {idx+1}:")
        print(f"All answers from multipath: {all_answers}")



if __name__ == '__main__':

    main()
