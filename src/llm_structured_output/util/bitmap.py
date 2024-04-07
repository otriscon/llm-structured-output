"""
Utilities to use the bitmap of accepted token ids returned by TokenAcceptor.
"""

from math import inf
from typing import Iterable


def count_set_bits(bitmap: int) -> int:
    """
    Count the number of bits set to one.
    """
    # FUTURE: self.ids.bit_count() available from Python 3.10 is said to be 6x faster
    return bin(bitmap).count("1")


def highest_bit_set(bitmap: int) -> int:
    """
    Return the highest bit set in the bitmap.
    """
    return bitmap.bit_length() - 1


def enumerate_set_bits(bitmap: int) -> Iterable[int]:
    """
    Generator that yields the indices of the set bits in the bitmap.
    Note that it does so from highest to lowest.
    """
    while bitmap:
        highest_bit = highest_bit_set(bitmap)
        yield highest_bit
        bitmap -= 1 << highest_bit


def bias_logits(np, logits, accepted_token_bitmap):
    """
    Apply a -inf bias to tokens that will not be accepted.
    """
    highest_token = highest_bit_set(accepted_token_bitmap)
    match_count = count_set_bits(accepted_token_bitmap)
    # Check whether there's more tokens to be rejected or to be allowed, then do what's less work.
    if match_count <= highest_token / 2:
        indices = np.array([*enumerate_set_bits(accepted_token_bitmap)])
        array = np.full(logits.shape, -inf)
        array[indices] = 0
    else:
        rejected_token_bitmap = (1 << (highest_token + 1)) - 1 - accepted_token_bitmap
        indices = np.array([*enumerate_set_bits(rejected_token_bitmap)])
        array = np.full(logits.shape, 0)
        array[indices] = -inf
    return np.add(logits, array)
