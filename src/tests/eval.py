# pylint: disable=missing-function-docstring
"""
Run a function calling evaluation with the Fireworks AI dataset or similar
https://huggingface.co/datasets/fireworks-ai/function-calling-eval-dataset-v0
"""
import argparse
import json
import time

from deepdiff import DeepDiff

from examples.llm_schema import Model
from llm_structured_output.util.output import info, bold, inverse, debug


def run_eval_case(model, case, header, temp=None, seed=None, preemptive_batch_size=0):
    messages = case["prompt"]
    gold_completion = json.loads(case["completion"].partition("<functioncall>")[2])
    tools = json.loads(case["tools"])

    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "function_call": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "const",
                                "const": tool["function"]["name"],
                            },
                            "arguments": tool["function"]["parameters"],
                        },
                        "required": ["name", "arguments"],
                    }
                },
                "required": ["function_call"],
            }
            for tool in tools
        ]
    }

    info(f"{header} Starting generation...")
    content = ""
    prompt_tokens = 0
    completion_tokens = 0
    completion_time = 0
    start_time = time.time_ns()

    for result in model.completion(
        messages,
        schema=schema,
        max_tokens=4000,
        temp=temp,
        seed=seed,
        preemptive_batch_size=preemptive_batch_size,
        cache_prompt=True,
    ):
        if result["op"] == "evaluatedPrompt":
            prompt_tokens += result["token_count"]
            prompt_time = result["time_ms"]
        elif result["op"] == "generatedTokens":
            completion_tokens += result["token_count"]
            completion_time += result["time_ms"]
            content += result["text"]
            bold(result["text"], end="", flush=True)
        elif result["op"] == "stop":
            print()
        else:
            debug(f"{result=}")
            assert False

    total_time = (time.time_ns() - start_time) / 1e6
    prompt_tps = prompt_tokens / prompt_time * 1e3
    completion_tps = completion_tokens / completion_time * 1e3
    info(
        f"{header} {prompt_tokens=} {prompt_tps=:.02f} {completion_tokens=} {completion_tps=:.02f} {prompt_time=:.02f} {completion_time=:.02f} {total_time=:.02f}"
    )

    completion = json.loads(content)["function_call"]
    diff = DeepDiff(gold_completion, completion)
    if diff:
        inverse(f"{header} DIFF:", diff)
        return False
    else:
        info(f"{header} PASS")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Run a function calling evaluation with the Fireworks AI dataset or similar"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx_model",
        help="The path to the model weights and tokenizer",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="The path to the evaluation dataset",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Start at the given evaluation case number",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Limit the number of cases to run",
    )
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.0,
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    parser.add_argument(
        "--preemptive",
        type=int,
        default=0,
        help="If greater than zero, the maximum size of the batch for pre-emptive decoding",
    )
    args = parser.parse_args()

    info("Loading model...")
    model = Model()
    model.load(args.model_path)

    with open(args.dataset_path, encoding="utf-8") as dataset:
        cases = json.load(dataset)
        pass_count = 0
        fail_count = 0
        t0 = time.time_ns()
        if args.count:
            end_index = args.skip + args.count
        else:
            end_index = len(cases)
        for i, case in enumerate(cases[args.skip : end_index]):
            if run_eval_case(
                model,
                case,
                f"[{i+args.skip}]",
                temp=args.temp,
                seed=args.seed,
                preemptive_batch_size=args.preemptive,
            ):
                pass_count += 1
            else:
                fail_count += 1
        average_time = (time.time_ns() - t0) / 1e9 / (pass_count + fail_count)
        info(f"Totals: {pass_count=} {fail_count=} {average_time=:.02}s")


main()
