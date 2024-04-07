"""
Tokenizer utils.
"""

SPIECE_UNDERLINE = "▁"


def extract_vocabulary(tokenizer) -> tuple[dict[int, str], int]:
    """
    Extract the vocabulary and eos_token_id from a Huggingfaze PreTrainedTokenizer.
    """
    return dict(
        [
            (i, fragment.replace(SPIECE_UNDERLINE, " "))
            for fragment, i in tokenizer.vocab.items()
        ]
    ), tokenizer.eos_token_id
