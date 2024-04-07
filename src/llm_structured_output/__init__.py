"""
LLM structured output: constrain generation to a JSON schema.
"""
from .json_schema_acceptor import JsonSchemaAcceptor, JsonSchemaAcceptorDriver
from .json_acceptor import JsonAcceptor
from .util.bitmap import bias_logits
from .util.tokenization import extract_vocabulary
