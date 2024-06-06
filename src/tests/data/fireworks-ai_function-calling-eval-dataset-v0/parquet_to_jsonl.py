"""
Convert a fireworks function calling dataset from parquet to jsonl that can be
used by the evaluation scripts.

https://huggingface.co/datasets/fireworks-ai/function-calling-eval-dataset-v0
"""

import sys
import json
import pyarrow.parquet as pq

if len(sys.argv) < 2:
    print("Need path to parquet file.")
    sys.exit(1)
input_file = sys.argv[1]
data = pq.read_table(input_file).to_pydict()
prompts = data["prompt"]
completions = data["completion"]
tools = data["tools"]

output_file = input_file.replace(".parquet", ".jsonl")
if output_file == input_file:
    output_file += ".jsonl"

with open(output_file, mode="w", encoding="utf-8") as f:
    for i, prompt in enumerate(prompts):
        json.dump(
            {
                "prompt": prompt,
                "tools": json.loads(tools[i]),
                # The source dataset contains one gold completion per case, but we output an array
                # to support multiple gold answers down the line.
                "gold": [
                    {
                        "type": "function",
                        "function": json.loads(
                            completions[i].partition("<functioncall>")[2]
                        ),
                    }
                ],
                "options": {
                    "prompt_includes_schema": True,
                    # This dataset has only cases where one tool is invoked, and the prompt includes
                    # an example in which the output is not an array but a single tool call.
                    "single_tool": True,
                },
            },
            f,
        )
        f.write("\n")
