# pylint: disable=missing-class-docstring,missing-function-docstring
"""
Example of JSON schema decoding with MLX.
"""
import argparse
import json
import time
from math import inf
from operator import itemgetter
from typing import Iterable, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.utils import load

from llm_structured_output import JsonSchemaAcceptorDriver
from llm_structured_output.util.bitmap import (
    bias_logits,
    count_set_bits,
    enumerate_set_bits,
)
from llm_structured_output.util.output import info, bold, bolddim, debug
from llm_structured_output.util.tokenization import HuggingfaceTokenizerHelper


class RejectedCompletion(Exception):
    """
    It's rare, but sometimes we reach a state where it's not possible to
    advance the acceptor. For example, when closing a JSON string we get
    a higher probability for slanted quotes than straight ones and select
    the wrong token. At that point, the LLM will continue generating with
    the prior that the string is closed, but our acceptor will remain in
    the string-accepting state. This can indicate an issue with the
    tokenizer vocabulary passed to the acceptor, or a bug in the code
    used to decode tokens from the LLM. If none of these apply, check that
    the LLM is actually able to generate JSON, although most are.
    """


class Model:
    def __init__(self):
        mx.random.seed(0)
        self.model = None
        self.tokenizer = None
        self.vocabulary = None
        self.eos_id = None
        self.json_schema_acceptor_driver_factory = None
        self._cached_prompt = None
        self._cached_cache = None

    def load(self, model_path: str):
        """
        Load locally or download from Huggingface hub.
        """
        self.model, tokenizer = load(model_path)
        self.tokenizer = HuggingfaceTokenizerHelper(tokenizer)
        self.vocabulary, self.eos_id = self.tokenizer.extract_vocabulary()
        self.json_schema_acceptor_driver_factory = (
            JsonSchemaAcceptorDriver.driver_factory_for_model(
                self.vocabulary, self.eos_id
            )
        )

    def _run_model(self, inputs: mx.array, cache=None):
        """
        This is like the model's __call__() method as implemented in
        `mlx-examples/llms/mlx_lm/models/*.py`, except it is also able
        to apply a causal mask once a cache is in place.
        """
        model = self.model.model
        h = model.embed_tokens(inputs)

        mask = None
        T = h.shape[1]  # pylint: disable=invalid-name
        if T > 1:
            if cache is None:
                S = 0  # pylint: disable=invalid-name
            else:
                S = cache[0][0].shape[2]  # pylint: disable=invalid-name
            mask = nn.MultiHeadAttention.create_additive_causal_mask(S + T)
            mask = mask.split([S])[1].astype(h.dtype)

        if cache is None:
            cache = [None] * len(model.layers)

        for e, layer in enumerate(model.layers):
            h, cache[e] = layer(h, mask, cache[e])

        out = model.norm(h)
        return self.model.lm_head(out), cache

    def _decode(self, tokens):
        return self.tokenizer.no_strip_decode(tokens)

    def _evaluate_prompt(
        self, prompt: list[int], prior_prompt: list[int] = None, prior_cache=None
    ):
        if prior_prompt:
            i = 0
            for i, t in enumerate(prior_prompt):
                # We need to leave at least one token to evaluate because we don't
                # save the past logits.
                if i >= len(prompt) - 1 or prompt[i] != t:
                    break
            cache = [
                (
                    layer_key_cache[:, :, :i, :],
                    layer_value_cache[:, :, :i, :],
                )
                for layer_key_cache, layer_value_cache in prior_cache
            ]
            tokens = prompt[i:]
        else:
            tokens = prompt
            cache = None

        logits, cache = self._run_model(mx.array(tokens)[None], cache)
        return logits, cache

    def _debug_top_tokens(self, logits, count=10):
        token_logits = sorted(
            enumerate(logits.tolist()), key=itemgetter(1), reverse=True
        )
        top_tokens = [
            (self._decode([t]), p) for t, p in token_logits[:count] if p != -inf
        ]
        debug("TOP TOKENS:", top_tokens)

    def _sample(self, logits, temp: float = 0):
        if temp == 0:
            result = mx.argmax(logits, axis=-1)
        else:
            result = mx.random.categorical(logits * (1 / temp))
        return result.item()

    def _sample_with_bias(
        self, logits, temp: float = 0, token_acceptor=None, lazy_bias: bool = True
    ):
        if token_acceptor is None:
            return self._sample(logits, temp)

        if lazy_bias:
            token = self._sample(logits, temp)
            try:
                token_acceptor.advance_token(token)
                return token
            except JsonSchemaAcceptorDriver.TokenRejected:
                pass

        accepted_token_bitmap = token_acceptor.select_valid_tokens()
        if not accepted_token_bitmap:
            raise RejectedCompletion()
        token = self._sample(bias_logits(mx, logits, accepted_token_bitmap), temp)
        token_acceptor.advance_token(token)
        return token

    def generate_without_schema(self, logits, cache, temp: Optional[float] = 0.0):
        """
        For testing / comparison purposes.
        """
        while True:
            tokens = [self._sample(logits[0, -1, :], temp)]
            yield tokens
            if tokens[-1] == self.eos_id:
                break
            logits, cache = self.model(mx.array(tokens)[None], cache)

    def generate_with_schema(
        self, logits, cache, token_acceptor, temp: Optional[float] = 0.0
    ):
        while True:
            tokens = [self._sample_with_bias(logits[0, -1, :], temp, token_acceptor)]
            yield tokens
            if tokens[-1] == self.eos_id:
                break
            logits, cache = self.model(mx.array(tokens)[None], cache)

    def generate_with_preemptive_decoding(
        self,
        logits,
        cache,
        token_acceptor,
        temp: Optional[float] = 0.0,
        max_batch_size=5,
    ):
        """
        Try to generate faster by precomputing two tokens at a time when possible.
        If we know that the acceptor will only accept a small set of tokens after
        the current one, we can evaluate a batch with one entry per possible
        future token. Each entry in the batch contains the current token sampled,
        which we have to evaluate anyway, and a second token corresponding to one
        of the possible tokens that could be sampled from the output to the first
        token. We get back logits for both tokens for each item in the batch: the
        logits for the first token will be the same (as long as the model applies
        a causal mask), and we can sample those logits to select from which of the
        items in the batch we can select the second token.
        """
        # Sample token from prompt evaluation
        accepted_token_bitmap = token_acceptor.select_valid_tokens()
        first_token_logits = bias_logits(mx, logits[0, -1, :], accepted_token_bitmap)
        first_token = self._sample(first_token_logits, temp)
        tokens = [first_token]
        yield tokens
        token_acceptor.advance_token(first_token)
        accepted_token_bitmap = token_acceptor.select_valid_tokens()

        while True:
            last_token = tokens[-1]
            if count_set_bits(accepted_token_bitmap) in range(1, max_batch_size + 1):
                # If the number of possible follow-up tokens is small, submit for
                # evaluation a batch of 2-token continuations.
                batch = []
                for followup_token in enumerate_set_bits(accepted_token_bitmap):
                    batch.append([last_token, followup_token])
                # Re-shape the cache to match the input.
                cache = [
                    (
                        mx.concatenate([layer_key_cache] * len(batch)),
                        mx.concatenate([layer_value_cache] * len(batch)),
                    )
                    for layer_key_cache, layer_value_cache in cache
                ]
            else:  # Otherwise, submit the normal one-token continuation.
                batch = [[last_token]]

            logits, cache = self._run_model(mx.array(batch), cache)
            mx.eval(logits)

            first_token_logits = bias_logits(mx, logits[0, 0, :], accepted_token_bitmap)
            first_token = self._sample(first_token_logits, temp)
            tokens = [first_token]

            if first_token == self.eos_id:
                yield tokens
                break

            token_acceptor.advance_token(first_token)
            accepted_token_bitmap = token_acceptor.select_valid_tokens()
            if not accepted_token_bitmap:
                raise RejectedCompletion()

            # If we had submitted 2-token continuations, we can decode a second token
            if len(batch[0]) > 1:
                index = next(  # Find which of the second tokens was selected
                    i
                    for i, batch_item in enumerate(batch)
                    if batch_item[1] == first_token
                )
                second_token_logits = bias_logits(
                    mx, logits[index, 1, :], accepted_token_bitmap
                )
                second_token = self._sample(second_token_logits, temp)
                tokens.append(second_token)

                token_acceptor.advance_token(second_token)
                accepted_token_bitmap = token_acceptor.select_valid_tokens()

                # Select the accepted generation in the cache, restoring it to batch dimension 1.
                cache = [
                    (
                        layer_key_cache.split([index, index + 1])[1],
                        layer_value_cache.split([index, index + 1])[1],
                    )
                    for layer_key_cache, layer_value_cache in cache
                ]

            yield tokens

    def generate_with_preemptive_decoding_constant_batch(
        self,
        logits,
        cache,
        token_acceptor,
        temp: Optional[float] = 0.0,
        max_batch_size=5,
    ):
        """
        [Experimental]
        Same as generate_with_preemptive_decoding(), but keeping the batch size
        constant through the generation to have consistent matrix sizes.
        This is just for experimenting with performance.
        """
        # Sample token from prompt evaluation
        accepted_token_bitmap = token_acceptor.select_valid_tokens()
        first_token_logits = bias_logits(mx, logits[0, -1, :], accepted_token_bitmap)
        first_token = self._sample(first_token_logits, temp)
        tokens = [first_token]
        yield tokens
        token_acceptor.advance_token(first_token)
        accepted_token_bitmap = token_acceptor.select_valid_tokens()

        # Re-shape the cache to match the constant batch size.
        cache = [
            (
                mx.array([layer_key_cache[0]] * max_batch_size),
                mx.array([layer_value_cache[0]] * max_batch_size),
            )
            for layer_key_cache, layer_value_cache in cache
        ]

        while True:
            last_token = tokens[-1]
            if count_set_bits(accepted_token_bitmap) in range(1, max_batch_size + 1):
                # If the number of possible follow-up tokens is small, submit for
                # evaluation a batch of 2-token continuations.
                batch = []
                for followup_token in enumerate_set_bits(accepted_token_bitmap):
                    batch.append([last_token, followup_token])
            else:  # Otherwise, submit the normal one-token continuation.
                batch = [[last_token]]

            # Fill up to the batch capacity so that the matrices have the right shape.
            batch_array = mx.array(batch + [batch[-1]] * (max_batch_size - len(batch)))

            logits, cache = self._run_model(batch_array, cache)
            mx.eval(logits)

            first_token_logits = bias_logits(mx, logits[0, 0, :], accepted_token_bitmap)
            first_token = self._sample(first_token_logits, temp)
            tokens = [first_token]

            if first_token == self.eos_id:
                yield tokens
                break

            token_acceptor.advance_token(first_token)
            accepted_token_bitmap = token_acceptor.select_valid_tokens()
            if not accepted_token_bitmap:
                raise RejectedCompletion()

            # If we had submitted 2-token continuations, we can decode a second token
            if len(batch[0]) > 1:
                index = next(  # Find which of the second tokens was selected
                    i
                    for i, batch_item in enumerate(batch)
                    if batch_item[1] == first_token
                )
                second_token_logits = bias_logits(
                    mx, logits[index, 1, :], accepted_token_bitmap
                )
                second_token = self._sample(second_token_logits, temp)
                tokens.append(second_token)

                token_acceptor.advance_token(second_token)
                accepted_token_bitmap = token_acceptor.select_valid_tokens()

                # Select the accepted generation in the cache for the next round.
                cache = [
                    (
                        mx.array([layer_key_cache[index]] * max_batch_size),
                        mx.array([layer_value_cache[index]] * max_batch_size),
                    )
                    for layer_key_cache, layer_value_cache in cache
                ]

            yield tokens

    def _generate_tokens(
        self,
        generator: Iterable,
        max_tokens: int = 1000,
    ) -> Iterable:
        start_time = time.time_ns()
        token_count = 0

        for tokens in generator:
            token_count += len(tokens)

            try:
                eos_index = tokens.index(self.eos_id)
                tokens = tokens[0:eos_index]
            except ValueError:
                eos_index = -1

            if tokens:
                text = self._decode(tokens)
                yield {
                    "op": "generatedTokens",
                    "text": text,
                    "token_count": len(tokens),
                    "time_ms": (time.time_ns() - start_time) / 1e6,
                }

            if eos_index >= 0:
                yield {"op": "stop", "reason": "end"}
                return

            if token_count >= max_tokens:
                yield {"op": "stop", "reason": "max_tokens"}
                return

            start_time = time.time_ns()

        assert False

    def completion(
        self,
        prompt: Union[str, Iterable[dict[str, str]]],
        schema: dict,
        encapsulated: bool = False,
        max_tokens: int = 1000,
        temp: float = 0.0,
        seed: int = None,
        preemptive_batch_size: int = 0,
        cache_prompt: bool = False,
    ):
        if seed is not None:
            mx.random.seed(seed)

        start_time = time.time_ns()
        prompt_tokens = self.tokenizer.encode_prompt(prompt)
        logits, cache = self._evaluate_prompt(
            prompt_tokens, self._cached_prompt, self._cached_cache
        )
        if cache_prompt:
            self._cached_prompt = prompt_tokens
            self._cached_cache = cache
        # Eager eval to more accurately reflect the prompt evaluation time.
        mx.eval(logits)
        prompt_time = time.time_ns() - start_time
        yield {
            "op": "evaluatedPrompt",
            "prompt": prompt,
            "token_count": len(prompt_tokens),
            "time_ms": prompt_time / 1e6,
            "prompt_tps": len(prompt_tokens) / (prompt_time / 1e9),
        }

        if schema:
            token_acceptor = self.json_schema_acceptor_driver_factory(
                schema,
                is_encapsulated_json=encapsulated,
            )
            if preemptive_batch_size > 0:
                generator = self.generate_with_preemptive_decoding(
                    logits,
                    cache,
                    token_acceptor,
                    temp,
                    max_batch_size=preemptive_batch_size,
                )
            else:
                generator = self.generate_with_schema(
                    logits, cache, token_acceptor, temp
                )
        else:
            generator = self.generate_without_schema(logits, cache, temp)

        token_count = 0
        generation_time = 0
        for generation_result in self._generate_tokens(generator, max_tokens):
            if generation_result["op"] == "generatedTokens":
                token_count += generation_result["token_count"]
                generation_time += generation_result["time_ms"]
            elif generation_result["op"] == "stop":
                generation_result["token_count"] = token_count
                generation_result["time_ms"] = generation_time
                # This is slightly incorrect, because the first token is generated
                # from the prompt evaluation.
                generation_result["generation_tps"] = token_count / (
                    generation_time / 1e3
                )
            yield generation_result


def main():
    parser = argparse.ArgumentParser(
        description="LLM inference script with schema-constrained sampling"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx_model",
        help="The path to the model weights and tokenizer",
    )
    parser.add_argument(
        "--prompt",
        default="Once upon a midnight dreary",
        help="The message to be processed by the model",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.0,
    )
    parser.add_argument("--seed", type=int, default=None, help="The PRNG seed")
    parser.add_argument(
        "--repeat-prompt",
        action=argparse.BooleanOptionalAction,
        help="Print prompt before start of generation",
    )
    parser.add_argument(
        "--schema",
        help="A JSON schema to constrain the output.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--encapsulated",
        action=argparse.BooleanOptionalAction,
        help="Whether the LLM is expected to encapsulate the JSON within ```json and ```.",
    )
    parser.add_argument(
        "--preemptive",
        type=int,
        default=0,
        help="If greater than zero, the maximum size of the batch for pre-emptive decoding",
    )

    args = parser.parse_args()

    info("Loading model from disk.")
    model = Model()
    model.load(args.model_path)

    if args.schema is not None:
        schema = json.loads(args.schema)
        info("Using schema")
    else:
        schema = None
    info("Starting generation...")

    for result in model.completion(
        prompt=args.prompt,
        schema=schema,
        encapsulated=args.encapsulated,
        max_tokens=args.max_tokens,
        temp=args.temp,
        seed=args.seed,
        preemptive_batch_size=args.preemptive,
    ):
        if result["op"] == "evaluatedPrompt":
            prompt_token_count = result["token_count"]
            prompt_time = result["time_ms"]
            prompt_tps = result["prompt_tps"]
            if args.repeat_prompt:
                bolddim(result["prompt"], flush=True)
        elif result["op"] == "generatedTokens":
            bold(result["text"], end="", flush=True)
        elif result["op"] == "stop":
            end_reason = result["reason"]
            generated_token_count = result["token_count"]
            generation_time = result["time_ms"]
            generation_tps = result["generation_tps"]
        else:
            assert False

    print()
    info(f"End reason: {end_reason}")
    info(f"Tokens: prompt {prompt_token_count}, generation {generated_token_count}")
    info(f"Tokens per second: prompt {prompt_tps:.2f}, generation {generation_tps:.2f}")
    info(f"Total time: prompt {prompt_time:.2f}ms, generation {generation_time:.2f}ms")


if __name__ == "__main__":
    main()
