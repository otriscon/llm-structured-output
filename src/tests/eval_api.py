# pylint: disable=missing-function-docstring
"""
Run a tool use evaluation using an LLM with an OpenAI-like API.
"""
import argparse
import json
import time
import requests

from llm_structured_output.util.output import info, inverse, debug, warning

from .eval_report import eval_completion


def run_eval_case(
    api_url,
    api_key,
    model_name,
    case,
    header,
    temp=0,
    seed=0,
    stream=False,
    out=None,
):
    options = case.get("options", {})
    prompt_includes_schema = options.get("prompt_includes_schema", False)

    payload = {
        "model": model_name,
        "messages": case["prompt"],
        "tools": case["tools"],
        "tool_choice": "auto",
        "temperature": temp,
        "seed": seed,
    }
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    if prompt_includes_schema and "api.openai.com" not in api_url:
        # Non-standard option, should not be set for OpenAI API.
        payload["tool_options"] = {
            # Do not dump the schema again, since it's already in the prompt
            "no_prompt_steering": True,
        }

    info(f"{header} Sending API request...")
    start_time = time.time_ns()

    r = requests.post(
        f"{api_url}/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
        stream=stream,
    )
    if stream:
        response = None
        tool_calls = []
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data:"):
                warning("Expected all server-sent events to start with 'data:'")
            line = line[5:].strip()
            if line == "[DONE]":
                break
            message = json.loads(line)
            if response is None:
                response = message
            elif "usage" in message:
                response["usage"] = message["usage"]
                if not message["choices"]:
                    continue
            tool_deltas = message["choices"][0]["delta"].get("tool_calls", [])
            if len(tool_deltas) > 1:
                warning(
                    f"Expected updates for one tool_call at a time, got multiple: {tool_deltas=}"
                )
            if tool_deltas:
                tool_delta = tool_deltas[0]
                index = tool_delta["index"]
                argument_delta = tool_delta["function"]["arguments"]
                if index == len(tool_calls):
                    tool_calls.append(tool_delta)
                    tool_name = tool_delta["function"][
                        "name"
                    ]  # name may not be present in additional updates
                    debug(
                        f"[call #{index}]\nname: {tool_name}\narguments: {argument_delta}",
                        end="",
                    )
                elif index == len(tool_calls) - 1:
                    tool_calls[index]["function"]["arguments"] += argument_delta
                    debug(argument_delta, end="")
                else:
                    warning(
                        f"Unexpected tool_delta out of sequence: "
                        f"current_index={len(tool_calls)-1} {tool_delta=}"
                    )
        response["choices"] = [
            {"message": {"role": "assistant", "tool_calls": tool_calls}}
        ]
        debug()
    else:
        response = r.json()
        debug(response)

    total_time = (time.time_ns() - start_time) / 1e6
    prompt_tokens = response["usage"]["prompt_tokens"]
    completion_tokens = response["usage"]["completion_tokens"]
    info(f"{header} {prompt_tokens=} {completion_tokens=} {total_time=:.02f}")

    if out:
        json.dump(response, out)
        out.write("\n")

    diff = eval_completion(case, response)
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
        "--api-url",
        type=str,
        default="https://api.openai.com",
        help="The URL of the API server",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="The URL of the API server",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-4o",
        help="The name of the model to use",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=str,
        help="The path to the evaluation dataset (JSONL)",
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
        "--stream",
        help="Use streaming API.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--output-file",
        help="Write completions to JSONL file.",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    out = None
    if args.output_file:
        out = open(args.output_file, mode="w", encoding="utf-8")

    with open(args.dataset_path, encoding="utf-8") as dataset:
        if args.count:
            end_index = args.skip + args.count
        else:
            end_index = None
        pass_count = 0
        fail_count = 0
        t0 = time.time_ns()
        for i, line in enumerate(dataset.readlines()):
            if i < args.skip:
                continue
            if end_index is not None and i == end_index:
                break
            case = json.loads(line)
            if run_eval_case(
                args.api_url,
                args.api_key,
                args.model_name,
                case,
                f"[{i}]",
                temp=args.temp,
                seed=args.seed,
                stream=args.stream,
                out=out,
            ):
                pass_count += 1
            else:
                fail_count += 1
        average_time = (time.time_ns() - t0) / 1e9 / (pass_count + fail_count)
        info(f"Totals: {pass_count=} {fail_count=} {average_time=:.02}s")

    if out:
        out.close()


main()
