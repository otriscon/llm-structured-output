# LLM Structured Output: JSON Schema, Function Calling, Tools

This repository contains a library to constrain LLM generation to structured
output, such as function calling a.k.a. tool use.

We include examples of application implementations using the MLX library.

Differences with other approaches:

- "JSON mode": this library constrains output to be valid JSON, but goes
  beyond JSON mode in also enforcing a JSON schema. This enables much tighter
  steeing: specifying data types, property names, etc.

- GBNF translation: rather than converting the JSON schema to a formal grammar,
  we steer the output directly using the schema, which enables more flexible
  and deeper control with lower overhead. For example, expressing minimum and
  maximum array or string lengths in GBNF can lead to very large set of
  production rules, and certain JSON schema features are simply not possible.

- Fine-tuning: our approach is complementary to fine-tuning an LLM to produce
  structured output. While fine-tuning currently can enhance but not guarantee
  adherence to a schema, our system introduces strong guarantees on the output.

## Demo

Coming soon<sup>TM</sup>

## What's in the box

You'll find:

- A framefork and set of acceptors for constraining LLM output, which are
  application-independent.

- Reference implementations and examples using Apple's MLX library.

### Framework and JSON acceptors

- An acceptor/state machine framework which progresses all valid states of a
  given graph simultaneously. This minimizes the need for backtracking, which
  is expensive for LLMs as it would require re-computing past tokens. In this
  sense, the concept is similar to a chart parser or Earley-style recognizer
  and shares a similar motivation. In practice, it's quite different because
  we're dealing with token-level input. We implemented several optimizations
  to minimize combinatorial explosion: we use a trie to traverse the token
  vocabulary in logarithmic time, and collapse the trie branches when multiple
  options are equivalent. We also prune the chart by removing equivalent
  states arrived at by different paths. See [acceptor.py](src/llm_structured_output/acceptor.py).

- A JSON acceptor based on the framework above that accepts valid JSON. See
  [json_acceptor.py](src/llm_structured_output/json_acceptor.py).

- A JSON schema acceptor based on both items above that accepts valid JSON that
  conforms to a JSON schema. See [json_schema_acceptor.py](src/llm_structured_output/json_schema_acceptor.py).
  Please note that most but not all JSON schema directives are implemented.
  Please open an issue if one that you need is not.

### Reference implementation / examples

- An example of using the acceptors above to guide decoding in an LLM using
  Apple's MLX framework. See [llm_schema.py](src/examples/llm_schema.py).
  This example includes several decoding techniques, including pre-emptive evaluation,
  which is a way to use the acceptor to anticipate the tokens that can be generated
  according to the schema, and use that to evaluate two tokens at a time instead of
  one, sometimes leading to noticeable performance improvements.

- A server example that implements an OpenAI-compatible API including tools / function
  calling. Unlike [OpenAI's](https://platform.openai.com/docs/api-reference/chat/object),
  this implementation always generates valid JSON, and does not return hallucinated
  parameters not defined in your function schema (but it may still hallucinate their
  values). See [server.py](src/examples/server.py).

## Usage

### Run the examples on Apple hardware with MLX

Clone this repo:

```sh
git clone https://github.com/otriscon/llm-structured-output.git
cd llm-structured-output
```

Optional, but recommended: create and activate a virtual environment with your favorite tool of choice, e.g.

```sh
python -m venv .venv
source .venv/bin/activate
```

Move into the examples folder and install the requirements, then move back:

```sh
cd src/examples
pip install -r requirements.txt
cd ..
```

Run the llm_schema example:

```sh
MODEL=mistralai/Mistral-7B-Instruct-v0.2

LLM_PROMPT='[INST] Parse the following address into a JSON object: "27 Barrow St, New York, NY 10014". Your answer should be only a JSON object according to this schema: {"type": "object", "properties": {"streetNumber": {"type": "number"}, "streetName": {"type": "string"}, "city": {"type": {"string"}}, "state": {"type": "string"}, "zipCode": {"type": "number"}}}. Do not explain the result, just output it. Do not add any additional information. [/INST]'

LLM_SCHEMA='{"type": "object", "properties": {"streetNumber": {"type": "number"}, "streetName": {"type": "string"}, "city": {"type": "string"}, "state": {"type": "string"}, "zipCode": {"type": "number"}}}'

python3 -m examples.llm_schema --model-path $MODEL --prompt "$LLM_PROMPT" --schema "$LLM_SCHEMA" --max-tokens 1000 --repeat-prompt
```

Run the server example:

```sh
MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.2 uvicorn examples.server:app --port 8080 --reload
```


### Using the JSON schema acceptor in your project

Install in your project with `pip install llm-structured-output` and
use a `JsonSchemaAcceptorDriver` within your normal generation loop:

```python
from llm_structured_output import JsonSchemaAcceptorDriver, extract_vocabulary, bias_logits

# ...

# Load the model as usual.
model, tokenizer = load(model_path)

# Instantiate a token acceptor
vocabulary, eos_id = extract_vocabulary(tokenizer)
token_acceptor = JsonSchemaAcceptorDriver(schema, vocabulary, eos_id)

cache = None
tokens = tokenizer.encode(prompt)

while tokens[-1] != eos_id:

    # Evaluate the model as usual. 
    logits, cache = model(mx.array(tokens)[None], cache)

    # Set probability to -inf for invalid tokens.
    accepted_token_bitmap = token_acceptor.select_valid_tokens()
    logits = bias_logits(mx, logits[0, -1, :], accepted_token_bitmap)

    # Sample as usual.
    tokens = [sample(logits, temp)]
    text = tokenizer.decode(tokens)
    print(text, end="")

    # Advance the acceptor to the next state.
    token_acceptor.advance_token(text)
```

## A note about guarantees on the output

Constraining the output of an LLM to follow a schema doesn't magically make the
LLM great at producing output that solves a particular task.

If an LLM that is not prompted or fine-tuned correctly to solve the task, it
will produce syntactically valid output but the values inside won't necessarily
constitute a good solution. As with any other technique, proper LLM prompting
and/or n-shot examples are crucial to avoid getting nice-looking,
well-formatted, schema-compliant nonsense.

In particular, it's crucial to instruct the LLM regarding the desired output
format, including making the desired schema part of the prompt. Here's an
example of a prompt that includes the schema:

```
Parse the following address into a JSON object: "27 Barrow St, New York, NY 10014".
Your answer should be only a JSON object according to this schema: {"type": "object", "properties": {"streetNumber": {"type": "number"}, "streetName": {"type": "string"}, "city": {"type": {"string"}}, "state": {"type": "string"}, "zipCode": {"type": "number"}}}.
Do not explain the result, just output it. Do not add any additional information.
```

In order to give the LLM a scratch-pad prior to JSON generation for e.g.
chain-of-thought reasoning, we have included an option for the acceptor to kick in
only on output within a section delimited by the lines `` ```json `` and `` ``` ``,
with the prior output treated as free text. This is enabled with the `is_encapsulated_json`
option of the `JsonSchemaAcceptorDriver` constructor. Here's an example of a
prompt that produces encapsulated JSON:
```
Your mission is to parse the following address into a JSON object: "27 Barrow St, New York, NY 10014".
Your answer should be a JSON object according to this schema: {"type": "object", "properties": {"streetNumber": {"type": "number"}, "streetName": {"type": "string"}, "city": {"type": {"string"}}, "state": {"type": "string"}, "zipCode": {"type": "number"}}}.
First, think through the task step by step, and then output a JSON object wrapped between the lines ```json and ```.
```

## Testing

The library has been tested with the following datasets:

- [Fireworks.ai's function calling eval dataset](https://huggingface.co/datasets/fireworks-ai/function-calling-eval-dataset-v0/)

- [ALU.AI's table extraction](https://blog.alu.ai/tables-and-structured-data/) evaluation dataset (not yet open-source)

## Performance

Since we need to select the acceptable tokens prior to sampling, constraining
the output according to a schema introduces a delay for every token, which
depends on the complexity of the schema. On the other hand, since the output is
guaranteed to be valid JSON and to conform to the schema, it can reduce the
number of tokens generated and reduce or eliminate the number of retries
required to solve the task.

### Pre-emptive decoding experiment
As an experiment to improve performance, we implement the option to use
pre-emptive decoding: when the range of tokens that can be accepted after the
current one is small, as often happens with structured output, we submit to the
LLM a batch of two-token continuations where the first token is the one that
was to be evaluated anyway, and the second token in each item in the batch is
one of the possible continuations predicted according to the schema. We can
then sample two tokens instead of one.  We find that this approach can
occasionally produce considerable increases in token generation speed, but in
general it can also considerably slow it down, depending on model and
quantization. We are investigating whether this is an effect of Apple hardware
architecture, the MLX library, or our own implementation mistakes.

### Benchmarks

- The following tests were perfomed on an Apple Studio with an M2 Ultra (24 core)
with 192GB of RAM using MLX version 0.9.0.

- The results are the average of 5 runs on a simple data extraction task with a
127-token prompt.

- Pre-emptive decoding was tested in two different forms: with a constant batch
  size, where we always sent the same size matrices for evaluation, and variable-
  size batching, where we made the batch large or shorter depending on the numer
  of possible follow-up tokens.

<br>

| Mistral-7B-v0.2-Instruct (fp16) | Prompt tps | Generation tps | Generation tokens |
| --- | :-: | :-: | :-: |
| No schema | 305.82 | 34.76 | 321 |
| Schema | 307.00 |	31.70 | 42 |
| Pre-emptive constant batch =5 | 211.72 | 33.16 | 42 |
| Pre-emptive variable batch <=5 | 321.85 | 36.53  | 42 |


**Notes:**

- Pre-emptive decoding accelerates generation even over schemaless generation.

<br>
<br>

| Mistral-7B-v0.2-Instruct (q4) | Prompt tps | Generation tps | Generation tokens |
| --- | :-: | :-: | :-: |
| No schema | 487.19 | 86.36 | 137 |
| Schema | 487.83 | 67.60 | 42 |
| Pre-emptive constant batch =5 | 139.61 | 27.16 | 42 |
| Pre-emptive variable batch <=5 | 488.88 | 36.25 | 42 |

**Notes:**

- Pre-emptive decoding is vastly slower, with the only change being quantization (?!)

<br>
<br>

| Mixtral-8x7B-Instruct-v0.1 (fp16) | Prompt tps | Generation tps | Generation tokens |
| --- | :-: | :-: | :-: |
| No schema | 3.48 | 2.23 | 50 |
| Schema | 3.49 | 2.21 | 50 |
| Pre-emptive constant batch =5 |2.36 | 1.16 | 50 |
| Pre-emptive variable batch <=5 | 3.18 | 1.68 | 50 |

**Notes:**

- This is the only tested model that outputs schema-conforming output without a schema.

- Pre-emptive decoding is a lot slower again.

<br>
<br>

| Mixtral-8x7B-Instruct-v0.1 (q4) | Prompt tps | Generation tps | Generation tokens |
| --- | :-: | :-: | :-: |
| No schema | 15.02 | 32.21 | 165 |
| Schema | 14.94 | 23.75 | 50 |
| Pre-emptive constant batch =5 | 9.29 | 11.28 | 50 |
| Pre-emptive variable batch <=5 | 15.02 | 17.94 | 50 |

## Roadmap

- Investigate performance enhancements. Is batching slow because of our
  implementation or the MLX implementation or Apple hardware limitations (flops?)

- Extend JSON schema support as needed. Please, feel free to open an issue if
  you need a feature that not supported at the moment. Also open to implement
  additional schemas such as YAML and reference implementations for other LLMs.

- Add formal test cases.

- Reference implementation for the Transformers library.

- Port to C++ and reference implementation for llama.cpp
