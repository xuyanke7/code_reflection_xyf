import argparse
import os
import sys
import numpy as np
import openai
import jsonlines
from typing import Union, List, Dict
from utils import read_jsonl, read_jsonl_gz
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
import const

# 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
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "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"


def parserargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--root_dir", type=str,
                        help="The root logging directory", default="rootdata")
    parser.add_argument("--dataset_path", type=str,
                        help="The path to the benchmark dataset", default="rootdata")
    parser.add_argument("--strategy", type=str,default='general',
                        help="Strategy: `simple`, `reflexion`")
    parser.add_argument("--language", type=str, help="Strategy: `py` or `rs`",default="py")
    parser.add_argument(
        "--model", type=str,default="gpt-3.5-turbo", help="OpenAI models only for now. For best results, use GPT-4")
    parser.add_argument("--pass_at_k", type=int,
                        help="Pass@k metric", default=1)
    parser.add_argument("--max_iters", type=int,
                        help="The maximum number of self-improvement iterations", default=10)
    parser.add_argument("--expansion_factor", type=int,
                        help="The expansion factor for the reflexion UCS and A* strategy", default=3)

    # parser.add_argument("--is_leetcode", action='store_true',
    #                     help="To run the leetcode benchmark")  # Temporary

    # parser.add_argument("--verbose", action='store_true',
    #                     help="To print live logs")
    args = parser.parse_args()
    return args


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def gpt_chat(
    model: str,
    sys_msg: str,
    user_msg: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_comps=1,
) -> Union[List[str], str]:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=num_comps,
    )
    if num_comps == 1:
        return response.choices[0].message.content

    return [choice.message.content for choice in response.choices]


def code_general(dataset: List[dict],
                 pass_at_k: int,
                 log_path: str,
                 model: str = "gpt-3.5-turbo",) -> None:
    for i, item in enumerate(dataset):
        sys_instruction = item["prompt"]
        print('----------------------- SYSTEM MESSAGE -----------------------')
        print(const.PY_SIMPLE_CHAT_INSTRUCTION)
        print(' ----------------------- USER MESSAGE -----------------------')
        print(sys_instruction)
        res_main = gpt_chat(
            model=model,
            sys_msg=sys_instruction,
            user_msg=const.PY_SIMPLE_CHAT_INSTRUCTION,
        )
        print('----------------------- RESPONSE -----------------------')
        print(res_main)
        eval_general(item, res_main)
        break

def eval_general(dataitem, response):
    ...


def main(args):
    # check root dir
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    # check dataset
    args.dataset_path = "rootdata\humaneval-py_sample30.jsonl" # tmp
    dataset_name = os.path.basename(args.dataset_path).replace("jsonl", "")
    print(f"loading {dataset_name}")
    
    if args.dataset_path.endswith(".jsonl"):
        dataset = read_jsonl(args.dataset_path)
    else:
        raise ValueError(f"File `{args.dataset_path}` is not a jsonl file.")
    print(f"loaded {len(dataset)} items")

    # check logpath
    args.run_name = "test_general"
    log_dir = os.path.join(args.root_dir, args.run_name)
    log_path = os.path.join(
        log_dir, f"{dataset_name}_{args.strategy}_{args.max_iters}_{args.model}_pass_at_k_{args.pass_at_k}_{args.language}.jsonl")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(f"Logging to {log_path}")


    code_general(dataset, args.pass_at_k, log_path, args.model)


if __name__ == "__main__":
    args = parserargs()
    main(args)
