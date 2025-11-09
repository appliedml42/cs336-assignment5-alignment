import datetime
import glob
import json
import os
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, List

from datasets import load_dataset
from drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams

# CACHE ALL PROMPTS
PROMPT_DIR = "/workspace/cs336-assignment5-alignment/cs336_alignment/prompts"
PROMPTS = {}
for path in glob.glob(f"{PROMPT_DIR}/*.prompt"):
    path = Path(path)
    prompt_name = path.stem
    with open(path) as f:
        prompt = f.read()

    if prompt_name == "r1_zero":
        PROMPTS[prompt_name] = (prompt, "</answer>")


def setup_model(model_name: str, download_dir: str):
    return LLM(model=model_name, download_dir=download_dir)


def setup_dataset(path: str, name: str, split: str, cache_dir: str, preprocess):
    dataset = load_dataset(path, name=name, split=split, cache_dir=cache_dir)
    dataset = dataset.map(preprocess)
    return dataset


def r1_zero_prompt_preprocess(example):
    prompt, _ = PROMPTS["r1_zero"]
    prompt_with_question = prompt.format(question=example["question"])

    return {"prompt": prompt_with_question, "ground_truth": example["answer"]}


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    sampling_params: SamplingParams,
    run_dir: str,
):
    outputs = vllm_model.generate(prompts, sampling_params)
    report_data = []
    for output, ground_truth, prompt in zip(outputs, ground_truths, prompts):
        response = output.outputs[0].text
        reward = reward_fn(response=response, ground_truth=ground_truth, fast=False)
        report_data.append({"prompt": prompt, "response": response, "reward": reward})

    output_path = os.path.join(run_dir, "evals.jsonl")
    with open(output_path, "w") as writer:
        for data in report_data:
            json.dump(data, writer, ensure_ascii=False)
            writer.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a model on a given dataset.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="Name of the model to evaluate.",
        default="Qwen/Qwen2.5-Math-1.5B",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        default="openai/gsm8k",
        help="Path of the dataset for evaluation.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=False,
        default="main",
        help="Name of the dataset for evaluation.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        required=False,
        help="Split of the dataset for evaluation.",
    )
    parser.add_argument(
        "--download_dir", type=str, default=None, help="Hugging Face cache directory."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        default=None,
        help="Name of the prompt to use.",
    )
    parser.add_argument(
        "--reward_fn", required=True, default=None, help="Name of the reward function"
    )
    parser.add_argument("--temperature", type=float, default=1.0, required=False)
    parser.add_argument("--max_tokens", type=int, default=1024, required=False)
    parser.add_argument("--top_p", type=float, default=1.0, required=False)

    args = parser.parse_args()

    model_download_dir = os.path.join(args.download_dir, "models")
    model = setup_model(args.model_name, model_download_dir)

    preprocess = None
    prompt_specific_sampling_params = {}
    if args.prompt == "r1_zero":
        preprocess = r1_zero_prompt_preprocess
        _, stop = PROMPTS["r1_zero"]
        prompt_specific_sampling_params["stop"] = [stop]
    else:
        raise ValueError(f"{args.prompt} is not supported")

    dataset_download_dir = os.path.join(args.download_dir, "datasets")
    dataset = setup_dataset(
        args.dataset_path,
        args.dataset_name,
        args.dataset_split,
        dataset_download_dir,
        preprocess,
    )
    prompts = [x["prompt"] for x in dataset.to_list()]
    ground_truths = [x["ground_truth"] for x in dataset.to_list()]

    reward_fn = None
    reward_specific_sampling_params = {}
    if args.reward_fn == "r1_zero":
        reward_fn = r1_zero_reward_fn
        reward_specific_sampling_params["include_stop_str_in_output"] = True

    else:
        raise ValueError(f"Reward function {args.reward_fn} is not supported")

    prompt_specific_sampling_params.update(reward_specific_sampling_params)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        **prompt_specific_sampling_params
    )
    
    timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "-")
    run_dir = os.path.join(f"vllm-eval-run-{timestamp}")
    os.mkdir(run_dir)

    config_path = os.path.join(run_dir, "config.json")
    config = vars(args)
    config.update(prompt_specific_sampling_params)
    with open(config_path, "w") as writer:
        json.dump(config, writer, ensure_ascii=False, indent=4)
        
    
    evaluate_vllm(
        vllm_model=model,
        reward_fn=reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        sampling_params=sampling_params,
        run_dir=run_dir,
    )
