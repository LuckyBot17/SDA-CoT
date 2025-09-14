import argparse
import json
import random
import re
import sys
import time
import openai
import torch
import numpy as np

API_KEY = "<KEY>"
NO_SOLUTION = '-10086'


def create_dataloader(args)->list:
    set_random_seed(args.random_seed)
    questions, answers = load_data_uncertainty(args)
    dataset = []
    for idx in range(len(questions)):
        dataset.append({"question": questions[idx], "answer": answers[idx], "question_idx": idx})

    random.shuffle(dataset)
    print(f"dataloader size: {len(dataset)}")
    return dataset

def load_data_uncertainty(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "GSM8K":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1].replace(",", ""))
    if args.dataset == "CommonsenseQA":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])
    if args.dataset == "StrategyQA":
        if 'task' in args.dataset_path:
            with open(args.dataset_path) as f:
                json_data = json.load(f)["examples"]
                for line in json_data:
                    q = line["input"].strip()
                    a = int(line["target_scores"]["Yes"])
                    if a == 1:
                        a = "yes"
                    else:
                        a = "no"
                    questions.append(q)
                    answers.append(a)
    if args.dataset == "logiQA":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for item in json_data:
                options_str = ""
                for idx, opt in enumerate(item["options"]):
                    options_str += f"{chr(65 + idx)}. {opt} "
                q = f"{item['context'].strip()}\n{item['query'].strip()}\n{options_str.strip()}"
                a = item["options"][item["correct_option"]]
                questions.append(q)
                answers.append(a)
    print(f"dataset: {args.dataset}")
    print(f"dataset_size: {len(answers)}")
    args.dataset_size = len(answers)
    return questions, answers



def load_data(args):
    decoder = json.JSONDecoder()
    questions, answers, ids = [], [], []
    datapath = args.dataset_path
    test_start = 0
    test_end = 'full'
    if args.dataset == 'CommonsenseQA':
        datapath = 'dataset/CommonsenseQA/CommonsenseQA.jsonl'

    dataset_lower = args.dataset.lower()
    temp_id = lambda idx: f'temp_{idx}'

    def append_qa(q, a, id_):
        questions.append(q)
        answers.append(a)
        ids.append(id_)

    if dataset_lower in ['2wikimultihopqa', 'gsm8k', 'gsm8k_sorted', 'hotpotqa', 'loqiqa', 'strategyqa', 'svamp',
                         'svamp_sorted']:
        with open(datapath) as f:
            json_data = json.load(f)
            if dataset_lower == 'strategyqa':
                json_data = json_data["examples"]

            for idx, item in enumerate(json_data):
                if dataset_lower == 'svamp':
                    body = item['Body'].strip()
                    question = item['Question'].strip()
                    q = f"{body}. {question}" if not body.endswith('.') else f"{body} {question}"
                    a = float(item["Answer"])
                    id_ = item["ID"]
                elif args.dataset == 'svamp_sorted':
                    q = item['Question']
                    a = float(item['Answer'])
                    id_ = item['ID']
                elif dataset_lower == 'strategyqa':
                    q = item["input"].strip()
                    a = "yes" if int(item["target_scores"]["Yes"]) == 1 else "no"
                    id_ = temp_id(idx)
                elif dataset_lower in ['gsm8k', 'gsm8k_sorted']:
                    q = item['question']
                    a = float(item['answer'])
                    id_ = temp_id(idx)
                else:
                    raise ValueError(f'Not supported dataset: {args.dataset}')

                append_qa(q, a, id_)

    elif dataset_lower in ['commonsenseqa']:
        with open(datapath) as f:
            for idx, line in enumerate(f):
                json_res = decoder.raw_decode(line)[0]
                if dataset_lower == 'commonsenseqa':
                    choices = 'answer choices: ' + ''.join(
                        f" ({c['label']}) {c['text']}" for c in json_res["question"]["choices"]
                    )
                    q = f"{json_res['question']['stem'].strip()}"+ f"\n{choices}"
                    a = json_res["answerKey"]
                    id_ = temp_id(idx)
                else:
                    raise ValueError(f'Not supported dataset: {args.dataset}')

                append_qa(q, a, id_)

    elif dataset_lower in ['finqa', 'convfinqa']:
        with open(datapath) as f:
            json_data = json.load(f)
            for idx, item in enumerate(json_data):
                text = item['text'] + '\n'
                table = item['table'].strip() + '\n'
                if dataset_lower == 'convfinqa':
                    q = f'Question: {item["questions"]}\n'
                else:  # finqa
                    q = f'Question: {item["question"]}\n'
                a = item['answer']
                id_ = temp_id(idx)
                append_qa(text + table + q, a, id_)

    else:
        raise ValueError(f'Not supported dataset: {args.dataset}')

    if test_end == 'full':
        return questions[test_start:], answers[test_start:], ids[test_start:]
    else:
        return questions[test_start:test_end], answers[test_start:test_end], ids[test_start:test_end]

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_input_prompt(args, cot_flag:bool)->str:
    """
        构造few-shot cot提示
        """
    question, rationale, pred_ans = [], [], []

    with open(args.prompt_path, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["prompt"]
        for line in json_data:
            question.append(line["question"])
            rationale.append(line["rationale"])
            pred_ans.append(line["pred_ans"])

    index_list = list(range(len(question)))

    prompt_text = ""
    for i in index_list:
        if cot_flag:
            if args.dataset == "strategyqa":
                prompt_text += question[i] + " " + rationale[i] + " " + \
                               "So the answer is" + " " + pred_ans[i] + ".\n\n"
            else:
                prompt_text += question[i] + " " + rationale[i] + " " + \
                               args.direct_answer_trigger_for_fewshot + " " + pred_ans[i] + ".\n\n"
        else:
            prompt_text += question[i] + " " + args.direct_answer_trigger_for_fewshot + " " + pred_ans[i] + ".\n\n"
    return prompt_text

def GPT3_request(model:str, input_prompt:list, max_tokens:int, time_interval, temperature=0.7, stop=None):
    resp = None
    done = False
    while not done:
        try:
            openai.api_key = API_KEY
            resp = openai.Completion.create(
                model=model,
                prompt=input_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop = stop
            )
            done = True
        except:
            errno = sys.exc_info()[:2]
            if errno[0] == openai.error.InvalidRequestError:
                print(f"Invalid Request\nPrompt: {input_prompt}\n")
                print(f"Reason: {errno[1]}")
                assert False
            else:
                print(f"Error: {errno[0]}\n")
                print(f"Reason: {errno[1]}\n")
            # pause between each request to avoid rate limit
            time.sleep(time_interval)
    return resp

def select_topK_uncertainty(sort_key,args):
    # 1. 读取不确定性结果
    with open(f"{args.output_dir}/uncertainty_result_{args.dataset}_k{args.num_trials}.txt", 'r') as f:
        uncertainty_data = json.load(f)


    # 2. 根据sort_key排序并取前10
    uncertainty_data_sorted = sorted(uncertainty_data, key=lambda x: x[sort_key], reverse=True)
    top10 = uncertainty_data_sorted[:10]
    top10_indices = [item['dataset_idx'] for item in top10]

    # 3. 读取原始数据集
    csqa_path = args.dataset_path
    with open(csqa_path, 'r') as f:
        csqa_lines = f.readlines()

    # 4. 提取对应内容（dataset_idx即行为索引）
    top10_samples = []
    for idx in top10_indices:
        if idx < len(csqa_lines):
            sample = json.loads(csqa_lines[idx])
            top10_samples.append(sample)
        else:
            print(f'Warning: idx {idx} out of range!')

    # 5. 输出为json文件
    with open('inference_prompts/top10_uncertain_csqa.json', 'w') as f:
        json.dump(top10_samples, f, indent=2, ensure_ascii=False)

    print('已保存前十个高不确定性样本到 inference_prompts/top10_uncertain_csqa.json')


def extract_final_answer(args, answer):
    """
    从模型输出中提取最终答案

    Args:
        args: 参数对象
        answer: 模型生成的原始答案文本

    Returns:
        str: 提取出的最终答案
    """
    final_ans = ""

    if args.dataset == "CommonsenseQA":
        # 提取选项A/B/C/D/E
        matches = re.findall(r'A|B|C|D|E', answer)
        if matches:
            final_ans = matches[-1]

    elif args.dataset == "gsm8k":
        # 提取数值答案，移除逗号
        answer = answer.replace(",", "")
        matches = [s for s in re.findall(r'-?\d+\.?\d*', answer)]
        if matches:
            try:
                final_ans = str(round(float(matches[-1])))
            except:
                final_ans = ""

    elif args.dataset == "StrategyQA":
        # 提取yes/no答案
        answer = answer.lower()
        answer = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", answer)
        matches = [i for i in answer.split() if i in ("yes", "no")]
        if matches:
            final_ans = matches[-1]

    elif args.dataset in ("2wikimultihopQA", "hotpotQA"):
        # 提取最后一句话作为答案
        sentences = re.split(r'[.!?]', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            # 尝试找到"the answer is"后面的内容
            for sentence in reversed(sentences):
                match = re.search(r'(?:the answer is|therefore,?\s+(?:the answer is)?)\s+(.*?)$',
                                  sentence.lower(), re.IGNORECASE)
                if match:
                    final_ans = match.group(1).strip()
                    break
            # 如果没找到明确的答案标记，就用最后一句话
            if not final_ans and sentences:
                final_ans = sentences[-1]

    elif args.dataset == "logiQA":
        # 提取选项A/B/C/D
        matches = re.findall(r'A|B|C|D', answer)
        if matches:
            final_ans = matches[-1]

    return final_ans


def load_prompt(prompt_path):
    """
       只保留Question、rationale、pred_ans部分，拼接为prompt字符串返回。
       """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        prompt_list = data["prompt"]
        prompt_lines = []
        for item in prompt_list:
            q = item.get("question", "")
            rationale = item.get("rationale", "")
            pred_ans = item.get("pred_ans", "")
            # 拼接格式可根据需要调整，这里用空格分隔
            line = f"{q} {rationale}\n pred_answer:{pred_ans}"
            prompt_lines.append(line)
        prompt = "\n\n".join(prompt_lines)
    return prompt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        choices=["2wikimultihopQA", "GSM8K", "hotpotQA", "logiQA", "CommonsenseQA", "StrategyQA"],
                        help="dataset to inference")
    parser.add_argument("--selected_uncertainty_path", type=str,default="select_uncertainty_result/uncertainty_result_CommonsenseQA_k10.txt")
    parser.add_argument("--num_trials", type=int, default=10, help="number of trials to run for each question")
    parser.add_argument("--sort_by", type=str, choices=['disagreement', 'variance', 'entropy'],
                        help="sort the final result by given option")
    parser.add_argument("--output_dir", type=str, default="select_uncertainty_result", help="output directory")
    args = parser.parse_args()
    args.dataset = "CommonsenseQA"
    if args.dataset == "2wikimultihopQA":
        args.dataset_path = "dataset/2wikimultihopQA/2wiki.json"
    elif args.dataset == "GSM8K":
        args.dataset_path = "dataset/gsm8k/train.jsonl"
        args.prompt_path = "basic_cot_prompts/math_word_problems"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "hotpotQA":
        args.dataset_path = "dataset/hotpotqa/hotpot.json"
    elif args.dataset == "logiQA":
        args.dataset_path = "dataset/logiqa/logiQA.json"
    elif args.dataset == "CommonsenseQA":
        args.dataset_path = "dataset/CommonsenseQA/CommonsenseQA.jsonl"
        args.prompt_path = "basic_cot_prompts/csqa_problems"
        args.direct_answer_trigger = "\nSo the answer is"
    elif args.dataset == "StrategyQA":
        args.dataset_path = "dataset/StrategyQA/train.json"
        args.prompt_path = "basic_cot_prompts/strategyqa_problems"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    select_topK_uncertainty(sort_key='entropy',args=args)