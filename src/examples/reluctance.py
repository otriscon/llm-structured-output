# pylint: disable=missing-class-docstring,missing-function-docstring
"""
Example of JSON schema decoding with MLX.
"""
import argparse
import json

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load

from llm_structured_output import (
    JsonSchemaAcceptorDriver,
    HuggingfaceTokenizerHelper,
    bias_logits,
)
from llm_structured_output.util.output import info, setbg, setfg, clear


def compute_reluctance(logits, accepted_token_bitmap) -> float:
    """
    Sum the probabilities of each token that has higher probability than
    the highest-probability token selected by the schema. This gives an
    idea of the model's preference for tokens that don't follow the schema.
    """
    p = nn.softmax(logits)
    indices = mx.argsort(p)[::-1]
    r = 0
    for i in indices.tolist():
        if (1 << i) & accepted_token_bitmap:
            break
        r += p[i].item()
    return r


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LLM reluctance to generate according to the schema."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx_model",
        help="The path to the model weights and tokenizer",
    )
    parser.add_argument(
        "--schema",
        help="A JSON schema to constrain the output.",
        type=str,
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=1000,
        help="Maximum number of tokens to generate",
    )

    args = parser.parse_args()

    info("Loading model from disk...")
    model, tokenizer = load(args.model_path)
    schema = json.loads(args.schema)

    tokenizer_helper = HuggingfaceTokenizerHelper(tokenizer)
    vocabulary, eos_id = tokenizer_helper.extract_vocabulary()
    token_acceptor_factory = JsonSchemaAcceptorDriver.driver_factory_for_model(vocabulary, eos_id)
    token_acceptor = token_acceptor_factory(schema)


    info("Starting generation...")
    cache = None
    tokens = tokenizer_helper.encode_prompt(args.prompt)
    while tokens[-1] != eos_id:
        logits, cache = model(mx.array(tokens)[None], cache)
        accepted_token_bitmap = token_acceptor.select_valid_tokens()
        reluctance = compute_reluctance(logits[0, -1, :], accepted_token_bitmap)
        biased_logits = bias_logits(mx, logits[0, -1, :], accepted_token_bitmap)
        token = mx.argmax(biased_logits, axis=-1).item()
        if token == eos_id:
            break
        tokens = [token]
        text = tokenizer_helper.no_strip_decode(tokens)
        setbg(reluctance, 0.8 * (1 - reluctance), 0)
        setfg(1, 1, 1)
        print(text, end="")
        token_acceptor.advance_token(token)
    clear()
    print()


if __name__ == "__main__":
    main()
