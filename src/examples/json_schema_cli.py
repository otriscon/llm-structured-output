#!/usr/bin/env python3
"""
Command-line tool to validate a JSON input against a JSON schema.
"""

import argparse
import json
import sys

from llm_structured_output.json_schema_acceptor import JsonSchemaAcceptorDriver


def main():  # pylint: disable=missing-function-docstring
    arg_parser = argparse.ArgumentParser(
        description="""
      Incrementally validate a JSON input against a JSON schema.
    """,
    )
    arg_parser.add_argument(
        "schema",
        help='JSON schema string, or "@<filename>" file containing JSON schema, or "-" for stdin)',
    )
    arg_parser.add_argument(
        "json",
        help='JSON input string, or "@<filename>" file containing JSON input, or "-" for stdin)',
    )
    args = arg_parser.parse_args()

    if args.schema == "-":
        schema = sys.stdin
    elif args.schema[0] == "@":
        with open(args.schema[1:], encoding="utf-8") as f:
            schema = f.read()
    else:
        schema = args.schema
    schema = json.loads(schema)

    if args.json == "-":
        input_json = sys.stdin
    elif args.json[0] == "@":
        with open(args.json[1:], encoding="utf-8") as f:
            input_json = f.read()
    else:
        input_json = args.json

    # For test purposes, just split the input into groups of 3 letters.
    fragments = [input_json[i : i + 3] for i in range(0, len(input_json), 3)]
    eos_fragment = chr(3)
    eos_token = 0
    vocabulary = list(enumerate([eos_fragment] + [*set(fragments)]))
    reverse_vocabulary = dict((f, i) for i, f in vocabulary)
    tokens = [reverse_vocabulary[f] for f in fragments]

    acceptor_factory = JsonSchemaAcceptorDriver.driver_factory_for_model(vocabulary, eos_id=eos_token)
    acceptor = acceptor_factory(schema)
    fail = False
    for token in tokens:
        print(token, end="")
        try:
            acceptor.debug_advance_token(token)
        except acceptor.TokenRejected:
            fail = True
            break
    if not fail:
        print("<eos>")
        try:
            fail = acceptor.advance_token(eos_token)
        except acceptor.TokenRejected:
            fail = True
    if fail:
        print("[FAIL]")
        result = 1
    else:
        print("[SUCCESS]")
        result = 0
    print("\n".join(repr(c) for c in acceptor.cursors))
    return result


if __name__ == "__main__":
    sys.exit(main())
