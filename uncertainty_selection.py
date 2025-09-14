# This file used to generate uncertainty score for each question
from llama_model import llama_request, init_model
from extraction.answer_extraction import answer_extraction
from utils import *
import time
import argparse
import numpy as np
import json
from scipy.stats import entropy


def main():
    args = arg_parser()
    args.model_name = "llama"
    print('*****************************')
    print(f"dataset: {args.dataset}")
    print(f"model: {args.model}")
    print(f"method: {args.method}")
    print(f"qes_limit: {args.qes_limit}")
    print(f"num_trials:{args.num_trials}")
    print(f"sort_by: {args.sort_by}")
    print('*****************************')

    print(f"API_KEY: {API_KEY}")
    set_random_seed(args.random_seed)

    dataloader = create_dataloader(args)

    if args.dataset_size > 50:
        dataloader = dataloader[:50]  # only take 1000 questions randomly to annotate, randomness decided by seed
    print(f"Dataloader size: {len(dataloader)}")

    if args.qes_limit == 0:
        args.qes_limit = len(dataloader)

    start = time.time()
    result = create_uncertainty(args, dataloader)
    end = time.time()
    print('Total Execution Time: ', end - start, " seconds")

    # output the results
    path = f"{args.output_dir}/uncertainty_result_{args.dataset}_k{args.num_trials}.txt"
    with open(path, 'w') as f:
        try:
            f.write(json.dumps(result, indent=4))
        except:
            for item in result:
                try:
                    if args.dataset in ("gsm8k", "asdiv", "svamp", "singleeq", "addsub", "multiarith"):
                        f.write(f"{item}, uncertainty: {len(item[-1])}, variance: {item[1]}\n")
                    else:
                        f.write(f"{item}, uncertainty: {len(item[-1])}\n")
                except:
                    pass

    # 生成并保存前十个高不确定性样本
    select_topK_uncertainty(args.sort_by, args)

def generate_uncertainty_qes(args, question, model, tokenizer):
    """
    针对某一道题目，反复调用语言模型（例如 GPT-3），多次生成答案，然后统计这些答案的多样性，以此来衡量这道题的不确定性。
    """
    if args.method == "few_shot_cot":
        given_prompt = create_input_prompt(args, True)

    if args.dataset == "GSM8K":
        # the float is reserved for variance calculation result
        uncertainty_record = {'dataset_idx': question['question_idx'], 'variance': float, 'entropy': float,
                              'occurrence': {}}
    elif args.dataset == "StrategyQA":
        uncertainty_record = {'dataset_idx': question['question_idx'], 'entropy': float,
                              'occurrence': {"yes": 0, "no": 0}}
    else:
        uncertainty_record = {'dataset_idx': question['question_idx'], 'entropy': float, 'occurrence': {}}

    for trial in range(args.num_trials):
        # if zero-shot to generate uncertainty, construct first stage zero-shot extraction (step by step)
        if args.method == "few_shot_cot":
            prompt = given_prompt + "Q: " + question['question'] + "\nA: Let's think step by step."
        elif args.method == "zero_shot_cot":
            prompt = "Q: " + question['question'] + "\nA: Let's think step by step."
        prompt_list = [prompt]

        # if use zero-shot, here we get the first stage zero-shot result
        # if not use zero-shot, here we get the final output
        if "gpt" in args.model:
            responses = GPT3_request(model=args.model, input_prompt=prompt_list, max_tokens=args.max_length_cot,
                                     time_interval=args.api_time_interval
                                     , temperature=args.temperature, stop=['Question:', "Q:"])
        if "llama" in args.model:
            responses = llama_request(model, tokenizer, input_prompt=prompt_list, time_interval=args.api_time_interval,
                                      temperature=args.uncertainty_temperature)
        # construct second stage extraction, to generate a single arabic num answer
        if args.method == "zero_shot_cot":
            prompt_list[0] += responses['choices'][0]['text'] + args.direct_answer_trigger

            # get the second stage zero-shot rationale result -> arabic num answer
            responses = GPT3_request(model=args.model, input_prompt=prompt_list, max_tokens=args.max_length_cot,
                                     time_interval=args.api_time_interval,
                                     temperature=args.temperature, stop='.')

        # extract the pred answer
        pred_ans = answer_extraction(args, responses)

        # check uncertainty
        if pred_ans != "":
            if pred_ans in uncertainty_record['occurrence']:
                uncertainty_record['occurrence'][pred_ans] += 1  # increment answer occurrence
            else:
                uncertainty_record['occurrence'][pred_ans] = 1  # first occurence
        else:
            # Handle no solution case
            if NO_SOLUTION in uncertainty_record['occurrence']:
                uncertainty_record['occurrence'][NO_SOLUTION] += 1
            else:
                uncertainty_record['occurrence'][NO_SOLUTION] = 1

    # calculate the variance for the question (only applied to datasets with numerical answer)
    if args.dataset in ("gsm8k"):
        ans_list = []
        for ans, occurs in uncertainty_record['occurrence'].items():
            for i in range(int(occurs)):
                ans_list.append(float(ans))
        uncertainty_record['variance'] = np.var(ans_list)

    # calculate the entropy for all dataset
    frequency_list = list(uncertainty_record['occurrence'].values())
    uncertainty_record['entropy'] = entropy(frequency_list)

    # calculate the disagreement for all dataset
    uncertainty_record['disagreement'] = len(uncertainty_record['occurrence'])

    return uncertainty_record


# return a sorted list by uncertainty from high to low
def create_uncertainty(args, questions):
    """
    对一组问题批量计算不确定性分数，并根据指定方式对结果排序，最终返回排序后的不确定性结果列表。
    disagreement：按不同答案的数量排序（分歧度）。
    variance：按答案的方差排序（只对数值型数据集有效）。
    entropy：按答案分布的熵排序（混乱度）。
    """
    result = []
    count = 0
    model, tokenizer, device = init_model(args)
    for question in questions:
        if count == args.qes_limit:
            break
        uncertainty_record = generate_uncertainty_qes(args, question, model, tokenizer)
        print(f"The question id is {question['question_idx']}.It's uncertainty_record: {uncertainty_record}")
        result.append(uncertainty_record)
        count += 1

    if args.sort_by == "disagreement":  # 如果按照分歧度进行
        if args.dataset == "strategyqa":
            try:
                # sort based on the entropy or the difference between yes and no answers
                result.sort(key=lambda x: abs(x['occurrence']['yes'] - x['occurrence']['no']))
            except:
                # sort by disagreement
                result.sort(key=lambda x: -len(x['occurrence']))
        else:
            result.sort(key=lambda x: -len(x['occurrence']))
    elif args.sort_by == "variance" and args.dataset in ("gsm8k", "asdiv", "svamp", "singleeq", "addsub", "multiarith"):
        # sort by variance
        result.sort(key=lambda x: -x['variance'])
    elif args.sort_by == "entropy":
        result.sort(key=lambda x: -x['entropy'])
    return result


def arg_parser():
    parser = argparse.ArgumentParser(description="Uncertainty_Generation")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument("--dataset", type=str, choices=["2wikimultihopQA", "GSM8K", "hotpotQA", "logiQA", "CommonsenseQA", "StrategyQA"], help="dataset to inference")
    parser.add_argument("--prompt_path", type=str, default="./basic_cot_prompts/math_word_problems", help="prompts to use")
    parser.add_argument("--model", type=str, help="model used for decoding.")
    parser.add_argument("--model_path", type=str, help="your local model path")
    parser.add_argument("--method", type=str, choices=["zero_shot_cot", "few_shot_cot"], help="method" )
    parser.add_argument("--output_dir", type=str, default="select_uncertainty_result", help="output directory")
    parser.add_argument("--max_length_cot", type=int, default=256,help="maximum length of output tokens by model for reasoning extraction")
    parser.add_argument("--qes_limit", type=int, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing.")
    parser.add_argument("--api_time_interval", type=float, default=1.0, help="how many seconds sleep between each request")
    parser.add_argument("--temperature", type=float, default=0.7, help="")
    parser.add_argument("--num_trials", type=int, help="number of trials to run for each question")
    parser.add_argument("--sort_by", type=str, choices=['disagreement', 'variance', 'entropy'],help="sort the final result by given option")
    parser.add_argument("--concat_length", type=int, default=2, help='Used for task last_letters, indicates length of last letter concat')
    parser.add_argument("--uncertainty_temperature", type=float)
    args = parser.parse_args()
    args.dataset = "CommonsenseQA"
    args.model = "llama2-7b"
    args.model_path = "/root/autodl-tmp/meta-llama2-7b-chat/shakechen/Llama-2-7b-chat-hf"
    args.method = "few_shot_cot"
    args.uncertainty_temperature = 0.7
    args.qes_limit = 0
    args.num_trials = 10
    args.sort_by = 'disagreement'
    # Fill in the dataset path
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

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."

    return args


if __name__ == "__main__":
    llama_path = "./meta-llama2-7b-chat/shakechen/Llama-2-7b-chat-hf"
    main()