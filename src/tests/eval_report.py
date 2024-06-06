# pylint: disable=missing-function-docstring
"""
Create a markdown report from an evaluation dataset and one or more completions.
"""
import argparse
import json
import re
import sys

from deepdiff import DeepDiff


def eval_tool_calls(case, tool_calls):
    single_tool = case.get("options", {}).get("single_tool", False)

    best_diff_count = 1e10
    for gold_tool_calls in case["gold"]:
        if single_tool:
            # The gold set in the source dataset is a single tool invocation instead of an array.
            # We could use the legacy function_call method to force a single function call, but
            # we think it's better to evaluate the model for non-legacy tool use. If the model
            # comes up with multi-tool solutions that are deemed acceptable, we can then:
            # - Remove this flag for this evaluation case,
            # - Wrap each existing gold value for this case in an array,
            # - Add the new solution that has multiple invocations to the gold set for the case.
            gold_tool_calls = [gold_tool_calls]
        diff = DeepDiff(gold_tool_calls, tool_calls, verbose_level=2)
        if diff is None:
            best_diff = None
            best_diff_count = 0
            break
        else:
            diff_count = diff.get_stats()["DIFF COUNT"]
            if diff_count < best_diff_count:
                best_diff_count = diff_count
                best_diff = diff
    return best_diff


def eval_completion(case, completion):
    try:
        completion_tool_calls = completion["choices"][0]["message"]["tool_calls"]
    except (KeyError, TypeError) as e:
        sys.stderr.write(
            f"Completion object doesn't match expected format: {completion=}\n"
        )
        completion_tool_calls = [
            {
                "type": "error",
                "error": {
                    "error": f"Parsing tool_calls: {repr(e)}",
                    "completion_message": completion["choices"][0]["message"],
                },
            }
        ]

    # Remove call metadata (currently only id) to compare with gold.
    # Note that we expect the gold set in the evaluation dataset to have
    # deserialized function arguments rather than as a string.
    tool_calls = [
        (
            {
                "type": "function",
                "function": {
                    "name": tool_call["function"]["name"],
                    "arguments": json.loads(tool_call["function"]["arguments"]),
                },
            }
            if tool_call["type"] == "function"
            else {
                "type": tool_call["type"],
                tool_call["type"]: tool_call[tool_call["type"]],
            }
        )
        for tool_call in completion_tool_calls
    ]

    return eval_tool_calls(case, tool_calls)


CHANGE_FORMATTERS = {
    "type_changes": lambda path, change: f"_{path}_ ~~{repr(change['old_value'])} [{change['old_type'].__name__}]~~ {repr(change['new_value'])} [{change['new_type'].__name__}]",
    "values_changed": lambda path, change: f"_{path}_ ~~{repr(change['old_value'])}~~ {repr(change['new_value'])}",
    "dictionary_item_added": lambda path, change: f"➕ _{path}_ {repr(change)}",
    "dictionary_item_removed": lambda path, change: f"➖ ~~_{path}_ {repr(change)}~~",
    "iterable_item_added": lambda path, change: f"_{path}_ ➕ {repr(change)}",
    "iterable_item_removed": lambda path, change: f"_{path}_ ➖ ~~{repr(change)}~~",
    "set_item_added": lambda path, change: f"_{path}_ ➕ {repr(change)}",
    "set_item_removed": lambda path, change: f"_{path}_ ➖ ~~{repr(change)}~~",
}


def diff_to_md(diff):
    if not diff:
        return "✅"
    md_changes = []
    for change_type, changes in diff.items():
        formatter = CHANGE_FORMATTERS[change_type]
        for path, change in changes.items():
            path = re.sub(r"root\[(\d*)\]\['function'\]", "function_call[\\1].", path)
            path = re.sub(r"root\[(\d*)\]", "tool_call[\\1]", path)
            path = re.sub(r"\['arguments'\]\['([^']*)']", "\\1", path)
            md_changes.append(formatter(path, change))
    return " ⸱ ".join(md_changes)


def report_eval_case(
    case,
    completions,
    index,
    out,
):
    eval_diffs = [eval_completion(case, completion) for completion in completions]
    columns = [diff_to_md(diff) for diff in eval_diffs]
    out.write(f"{index} | {' | '.join(columns)}\n")
    results = [not diff for diff in eval_diffs]
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run a function calling evaluation with the Fireworks AI dataset or similar"
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=str,
        help="The path to the evaluation dataset (JSONL)",
    )
    parser.add_argument(
        "completions",
        metavar="completion_files",
        type=str,
        nargs="+",
        help="One or more jsonl files with completions for the evaluation dataset",
    )
    parser.add_argument(
        "--output-file",
        help="Write report to a file instead of stdout",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    input_files = [open(filename, encoding="utf-8") for filename in args.completions]

    out = sys.stdout
    if args.output_file:
        out = open(args.output_file, mode="w", encoding="utf-8")

    i = 0
    with open(args.dataset_path, encoding="utf-8") as dataset:
        for i, line in enumerate(dataset.readlines()):
            case = json.loads(line)
            completions = [
                json.loads(input_file.readline()) for input_file in input_files
            ]
            if i == 0:
                sum_results = [0 for completion in completions]
                models = [completion["model"] for completion in completions]
                out.write(f"case | {' | '.join(models)}\n")
                out.write(f"--- | {' | '.join(['---'] * len(models))}\n")
            results = report_eval_case(case, completions, i, out)
            sum_results = [sum_results[i] + result for i, result in enumerate(results)]
    total = i + 1
    out.write(
        f"pass | {' | '.join([f'{r} ({round(100*r/total, 2)}%)' for r in sum_results])}\n"
    )

    for input_file in input_files:
        input_file.close()
    if out:
        out.close()


if __name__ == "__main__":
    main()
