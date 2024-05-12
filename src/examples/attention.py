# pylint: disable=missing-function-docstring,missing-class-docstring
"""
Example server to visualize the generation mechanism.
"""
import json
from operator import itemgetter
import os
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load
from llm_structured_output import (
    JsonSchemaAcceptorDriver,
    HuggingfaceTokenizerHelper,
)
from llm_structured_output.util.output import info, warning

from .reusable_kv_cache import ReusableKVCache


def calc_prompt_perplexity(logits, prompt: list[int]):
    """
    Try to get a measure for how much a prompt is baked into the training of the LLM.
    When evaluating a prompt, we pass several (perhaps many) input tokens. The LLM
    returns a matrix with the logits for each next token in the sequence, applying a
    mask to avoid "seeing" tokens that appear after the current one. We can compare
    the probability distrubtion formed by these logits with the actual token that
    follows, giving us an idea of how "surprised" the model is by the next token in
    the prompt. Note that this is related but not quite the same as the perplexity
    metric used to measure the training quality of a language model.
    The output is a list with a value for each token in the prompt, with the value
    being zero for no suprise (the model assigns probability 1 to that token appearing
    at that position given the prior tokens), and tending to infinity for the model
    assigning zero probability for that token in that position. By convention, we
    assign a value of zero to the first token in the prompt.
    """
    # Input:
    #   batch_size, ntokens, voc_size = logits.shape
    #   len(prompt) == ntokens
    # Note that row i of the output logits vector corresponds to the evaluation after
    # input token i, i.e. it's the probability distribution for token i+1. The last
    # row of logits corresponds to the first token after the prompt.
    target = mx.array([prompt[1:]])
    loss = nn.losses.cross_entropy(logits[:, :-1, :], target)[0]
    # Add a zero for the first token in the prompt.
    return [0] + loss.tolist()


class ObservedLLM:
    def __init__(self, model_path: str):
        self.model, self.tokenizer = load(model_path)
        self.tokenizer_helper = HuggingfaceTokenizerHelper(self.tokenizer)
        self.vocabulary, self.eos_id = self.tokenizer_helper.extract_vocabulary()
        self.token_acceptor_factory = JsonSchemaAcceptorDriver.driver_factory_for_model(
            self.vocabulary, self.eos_id
        )
        self.cache = ReusableKVCache.for_model(self.model)
        self.tokens = []
        self.fragments = []
        self.layer_attention_scores = []
        self.token_acceptor = None
        self.layer_attention_scores = []

        # Replace the attention dot product function to be able to look into it.
        def mock_fast_scaled_dot_product_attention(
            queries, keys, values, *, scale, mask=None, stream=None
        ):
            """
            O = softmax(Q @ K.T, dim=-1) @ V
            """
            B, n_kv_heads, L, _ = keys.shape
            _, n_heads, _, _ = queries.shape
            repeats = n_heads // n_kv_heads

            def repeat(a):
                a = mx.concatenate([mx.expand_dims(a, 2)] * repeats, axis=2)
                return a.reshape([B, n_heads, L, -1])

            keys, values = map(repeat, (keys, values))

            scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
            if mask is not None:
                scores += mask
            scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
            result = scores @ values
            self.layer_attention_scores.append(scores[0, :, -1, :].tolist())
            return result

        mx.fast.scaled_dot_product_attention = mock_fast_scaled_dot_product_attention

    def start(self, prompt: str, schema: dict):
        if schema is None:
            self.token_acceptor = None
        else:
            self.token_acceptor = self.token_acceptor_factory(schema)

        prior_tokens = self.tokens
        self.tokens = self.tokenizer_helper.encode_prompt(prompt)
        self.fragments = [
            self.tokenizer_helper.no_strip_decode([token]) for token in self.tokens
        ]

        # If we had started a generation before, try to reuse as much of the cache as possible.
        i = 0
        for i, t in enumerate(prior_tokens):
            if i >= len(self.tokens) - 1 or self.tokens[i] != t:
                break
        for layer_cache in self.cache:
            layer_cache.reuse(len(self.tokens), i)
        new_tokens = self.tokens[i:]

        print(f"{new_tokens}")
        return self._generate(new_tokens)

    def add_token(self, token):
        self.tokens.append(token)
        self.fragments.append(self.tokenizer_helper.no_strip_decode([token]))
        if self.token_acceptor:
            self.token_acceptor.advance_token(token)
        return self._generate([token])

    def _generate(self, new_input_tokens: list[int]):
        self.layer_attention_scores = []
        logits = self.model(mx.array(new_input_tokens)[None], self.cache)

        TOP_TOKEN_COUNT = 1000
        probs = mx.softmax(logits[0, -1, :])
        top_token_partition = mx.argpartition(probs, -TOP_TOKEN_COUNT)[
            -TOP_TOKEN_COUNT:
        ]
        top_token_probs = sorted(
            [*zip(top_token_partition.tolist(), probs[top_token_partition].tolist())],
            key=itemgetter(1),
            reverse=True,
        )

        if len(new_input_tokens) > 1:
            prompt_perplexity = calc_prompt_perplexity(logits, new_input_tokens)
        else:
            prompt_perplexity = None

        if self.token_acceptor:
            accepted_token_bitmap = self.token_acceptor.select_valid_tokens()
            rejected_top_tokens = set(
                token
                for token, _ in top_token_probs
                if not (accepted_token_bitmap & (1 << token))
            )
        else:
            rejected_top_tokens = set()

        top_tokens = [
            (
                token,
                self.tokenizer_helper.no_strip_decode([token]),
                p,
                token in rejected_top_tokens,
            )
            for token, p in top_token_probs
        ]

        return {
            "attention_scores": self.layer_attention_scores,
            "fragments": self.fragments,
            "top_tokens": top_tokens,
            "prompt_perplexity": prompt_perplexity,
        }


try:
    MODEL_PATH = os.environ["MODEL_PATH"]
except KeyError:
    MODEL_PATH = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
    info("No MODEL_PATH environment variable, using default model.")

info(f"Loading model {MODEL_PATH}...")
llm = ObservedLLM(MODEL_PATH)


app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    exc_str = f"{exc}"
    warning(f"RequestValidationError: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


@app.get("/")
def get_root():
    return FileResponse(
        f"{os.path.dirname(os.path.realpath(__file__))}/static/attention.html"
    )


@app.get("/status")
def get_status():
    return {"status": "OK"}


class GenerationStartRequest(BaseModel):
    prompt: str
    schema: Optional[str] = None


@app.post("/generation/start")
async def post_generation_start(request: GenerationStartRequest):
    if request.schema:
        schema = json.loads(request.schema)
    else:
        schema = None
    response = llm.start(request.prompt, schema)
    return response


class GenerationTokenRequest(BaseModel):
    token: int


@app.post("/generation/token")
async def post_generation_token(request: GenerationTokenRequest):
    response = llm.add_token(request.token)
    return response
