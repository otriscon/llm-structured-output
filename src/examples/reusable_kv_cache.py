"""
Helper with improvements over mlx-lm's KVCache.
"""

import mlx.core as mx
from mlx_lm.models.cache import KVCache


class ReusableKVCache(KVCache):
    """
    Usability improvements over KVCache.
    """

    @classmethod
    def for_model(cls, model):
        return [cls() for _ in model.layers]

    def reuse(self, new_prompt_length, common_prefix_length):
        """
        Reuse (part of) this cache for a new prompt that shares a prefix with it.
        """
        if self.keys is None:
            return
        # Clip the cache to the common length.
        self.offset = common_prefix_length
        # Make sure the cache can fit the whole prompt. Because the offset is
        # (very likely) not a multiple of the step size, update_and_fetch()
        # won't resize the cache when evaluating the rest of the prompt as it
        # would if it were an empty cache.
        current_size = self.keys.shape[2]
        if current_size < new_prompt_length:
            _, n_kv_heads, _, k_head_dim = self.keys.shape
            v_head_dim = self.values.shape[3]
            n_steps = (self.step + new_prompt_length - 1) // self.step
            k_add_shape = (1, n_kv_heads, n_steps * self.step - current_size, k_head_dim)
            v_add_shape = (1, n_kv_heads, n_steps * self.step - current_size, v_head_dim)
            k_zeros = mx.zeros(k_add_shape, self.keys.dtype)
            v_zeros = mx.zeros(v_add_shape, self.values.dtype)
            self.keys = mx.concatenate([self.keys, k_zeros], axis=2)
            self.values = mx.concatenate([self.values, v_zeros], axis=2)

    def update_and_fetch(self, keys, values):
        """
        Override the base class method to allow the cache to be used with batches of
        size greater than 1.
        This is just a tiny change in the line that determines the shape.
        """
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
    
        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

