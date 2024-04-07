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


def run_eval_case(model, case, header):
    messages = case["prompt"]
    gold_completion = json.loads(case["completion"].partition("<functioncall>")[2])
    tools = json.loads(case["tools"])

    prompt = "\n".join(
        (
            message.content
            if message["role"] == "assistant"
            else f"[INST] {message['content']} [/INST]"
        )
        for message in messages
    )

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
            }
            for tool in tools
        ]
    }

    info(f"{header} Starting generation...")
    content = ""
    prompt_tokens = 0
    completion_tokens = 0
    completion_time = 0

    for result in model.completion(
        prompt,
        schema=schema,
        max_tokens=4000,
        temp=0,
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

    prompt_tps = prompt_tokens / prompt_time * 1e3
    completion_tps = completion_tokens / completion_time * 1e3
    info(
        f"{header} {prompt_tokens=} {prompt_tps=:.02f} {completion_tokens=} {completion_tps=:.02f}"
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
    args = parser.parse_args()

    info("Loading model...")
    model = Model()
    model.load(args.model_path)

    with open(args.dataset_path, encoding="utf-8") as dataset:
        cases = json.load(dataset)
        pass_count = 0
        fail_count = 0
        t0 = time.time_ns()
        for i, case in enumerate(cases[args.skip:]):
            if run_eval_case(model, case, f"[{i+args.skip}]"):
                pass_count += 1
            else:
                fail_count += 1
        average_time = (time.time_ns() - t0) / 1e9 / (pass_count + fail_count)
        info(f"Totals: {pass_count=} {fail_count=} {average_time=:.02}s")


main()
